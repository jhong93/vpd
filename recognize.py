#!/usr/bin/env python3

import os
import argparse
import csv
from collections import defaultdict, Counter
from tabulate import tabulate
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
torch.set_num_threads(8)

from util.io import load_json, store_pickle, load_pickle
from util.eval import save_confusion_matrix
from util.video import get_metadata
from util.classifier import BaseSeqModel
from util.neighbors import KNearestNeighbors, Neighbors, build_dtw_distance_fn
from action_dataset.load import load_action_ids

import finegym.util as finegym
import diving48.util as diving48
import video_dataset_paths as dataset_paths

KNN_MODELS = ['dtw']
SEQ_MODELS = ['lstm', 'gru', 'cnn']

DEFAULT_NUM_EPOCHS = 500
DIVING48_FULL_NUM_EPOCHS = 200
DIVING48_LOW_SHOT_NUM_EPOCHS = 500

DATASETS = ['fx', 'diving48', 'diving48v1', 'tennis', 'fs']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_dir', type=str)
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=DATASETS)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('--algorithm', type=str, default='gru',
                        choices=KNN_MODELS + SEQ_MODELS)

    parser.add_argument('--retrieve', action='store_true',
                        help='Action retrieval, instead of recognition')

    parser.add_argument('-ne', '--num_train_examples', nargs='+',
                        type=int, default=[-1])

    parser.add_argument('-k', type=int, default=1, help='For k-NN only')

    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--target_fps', type=int, default=25)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--attn', action='store_true')

    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('-vf', '--val_freq', type=int, default=10)

    parser.add_argument('-nt', '--n_trials', type=int, default=1)
    parser.add_argument('-ntf', '--no_test_flip', action='store_true')

    parser.add_argument('-w', '--load_weights', type=str)
    return parser.parse_args()


class SeqModel:

    def __init__(self, arch_type, train_embs, train_labels, hidden_dim,
                 val_embs=None, val_labels=None, load_weights=None,
                 **kwargs):
        # The classes present may not be contiguous
        classes = Counter([train_labels[seq] for seq in train_embs])
        self.classes = list(sorted(classes.keys()))
        self.top_class = classes.most_common()[0][0]

        def make_dataset(all_embs, labels):
            def get_target(seq):
                return self.classes.index(labels[seq])

            X, y = [], []
            for i, (seq, embs) in enumerate(all_embs.items()):
                if embs is not None:
                    tgt = get_target(seq)
                    if len(embs.shape) == 3:
                        for j in range(embs.shape[1]):
                            X.append(embs[:, j, :])
                            y.append(tgt)
                    else:
                        X.append(embs)
                        y.append(tgt)
            return X, np.array(y)

        X, y = make_dataset(train_embs, train_labels)
        X_val, y_val = make_dataset(val_embs, val_labels)

        self.model = BaseSeqModel(
            arch_type, X, y, hidden_dim, X_val=X_val, y_val=y_val,
            load_weights=load_weights, **kwargs)

    def predict(self, x, ensemble=True):
        if x is not None:
            try:
                if len(x.shape) == 3:
                    if ensemble:
                        pred = self.model.predict_n(
                            *[x[:, i, :] for i in range(x.shape[1])]
                        )[0]
                    else:
                        pred = self.model.predict(x[:, 0, :])[0]
                else:
                    pred = self.model.predict(x)[0]
                return self.classes[pred], None
            except Exception as e:
                print(e)

        # Prevent null predictions
        return self.top_class, None

    def save_model(self, out_path):
        self.model.save(out_path)


class KnnModel:

    def __init__(self, dist_type, train_embs, train_labels, k):
        # The classes present may not be contiguous
        classes = Counter([train_labels[seq] for seq in train_embs])
        self.top_class = classes.most_common()[0][0]

        dist_fn2 = None
        if dist_type == 'dtw':
            dist_fn = build_dtw_distance_fn('symmetricP2')
            dist_fn2 = build_dtw_distance_fn('symmetric2')
        else:
            raise NotImplementedError()

        X, y, val = [], [], []
        for seq, embs in train_embs.items():
            tgt = train_labels[seq]
            if len(embs.shape) == 3:
                for i in range(embs.shape[1]):
                    X.append(embs[:, i, :])
                    y.append(tgt)
                    val.append(seq)
            else:
                X.append(embs)
                y.append(tgt)
                val.append(seq)
        self.val = val
        self.model = KNearestNeighbors(X, y, dist_fn, k=k)
        self.model2 = None
        if dist_fn2 is not None:
            self.model2 = KNearestNeighbors(X, y, dist_fn2, k=k)

    def predict(self, x, ensemble=True):
        def _predict(model):
            if len(x.shape) == 3:
                if ensemble:
                    pred, i = model.predict_n(
                        *[x[:, j, :] for j in range(x.shape[1])])
                else:
                    pred, i = model.predict(x[:, 0, :])
            else:
                pred, i = model.predict(x)
            if i is None:
                raise ValueError('No prediction')
            return pred, self.val[i]

        if x is not None:
            try:
                return _predict(self.model)
            except Exception as e:
                if self.model2 is not None:
                    try:
                        return _predict(self.model2)
                    except Exception as e:
                        print(e)
                print('Failed to predict')
        return self.top_class, None

    def save_model(*args):
        print('Nothing to save for KNN')


def sample_embeddings(embs, labels, n, keep_ratio=False):
    label_to_seqs = defaultdict(list)
    for seq in embs:
        label_to_seqs[labels[seq]].append(seq)
    least_common_count = min(len(x) for x in label_to_seqs.values())

    sub_seqs = []
    for seqs in label_to_seqs.values():
        tmp = round(len(seqs) / least_common_count * n) if keep_ratio else n
        if len(seqs) > tmp:
            seqs = np.random.choice(seqs, tmp, replace=False)
        sub_seqs.extend(seqs)
    return {s: embs[s] for s in sub_seqs}


def cache_video_meta_dict(path, meta_dict):
    store_pickle(path, meta_dict)


def load_finegym_data(dataset, emb_dir, norm, target_fps):
    if dataset == 'fx':
        video_dir = dataset_paths.FX_VIDEO_DIR
    else:
        raise NotImplementedError(dataset)

    video_meta_cache = 'data/sports.cache/{}.video_meta.pkl'.format(dataset)
    if os.path.isdir(video_dir):
        video_meta_dict = {
            os.path.splitext(v)[0]: get_metadata(os.path.join(video_dir, v))
            for v in tqdm(os.listdir(video_dir), desc='Loading video metadata')
            if v.endswith('.mp4')
        }
        # cache_video_meta_dict(video_meta_cache, video_meta_dict)
    else:
        print('Raw videos not found! Using cached metadata.')
        video_meta_dict = load_pickle(video_meta_cache)

    # These contain labels for all datasets. We use the videos in video_dir
    # to restrict the evaluation to only floor exercise
    annotations = load_json(finegym.ANNOTATION_FILE)
    categories = finegym.load_categories(finegym.GYM99_CATEGORY_FILE)
    train_labels = finegym.load_labels(finegym.GYM99_TRAIN_FILE)
    test_labels = finegym.load_labels(finegym.GYM99_VAL_FILE)

    label_kwargs = {'pre_seconds': 0.25, 'target_fps': target_fps,
                    'emb_dir': emb_dir, 'norm': norm}

    train_actions = finegym.load_actions(
        annotations, tqdm(train_labels, desc='Loading train embeddings'),
        video_meta_dict, **label_kwargs)
    train_embs = {k: v[1] for k, v in train_actions.items()}

    test_actions = finegym.load_actions(
        annotations, tqdm(test_labels, desc='Loading test embeddings'),
        video_meta_dict, **label_kwargs)
    test_embs = {k: v[1] for k, v in test_actions.items()}

    def get_intervals(actions):
        video_interval_dict = defaultdict(list)
        for a in actions:
            video, start_end = a.split('_A_')
            start, end = [int(x) for x in start_end.split('_')]
            video_interval_dict[video].append((start, end))
        return video_interval_dict

    train_video_dict = get_intervals(train_embs.keys())
    test_video_dict = get_intervals(test_embs.keys())
    print('Train videos:', len(train_video_dict))
    print('Test videos:', len(test_video_dict))
    print('Train/test video overlap:',
          len(set(train_video_dict) & set(test_video_dict)))
    for a in test_video_dict:
        train_video_dict[a].extend(test_video_dict[a])
    return (categories, train_embs, train_labels, test_embs, test_labels,
            train_video_dict)


def load_diving48_data(emb_dir, norm, target_fps, use_v1):
    video_meta_cache = 'data/sports.cache/diving48.video_meta.pkl'
    if os.path.isdir(dataset_paths.DIVING48_VIDEO_DIR):
        video_meta_dict = {
            os.path.splitext(v)[0]: get_metadata(
                os.path.join(dataset_paths.DIVING48_VIDEO_DIR, v))
            for v in tqdm(os.listdir(dataset_paths.DIVING48_VIDEO_DIR),
                        desc='Loading video metadata')
            if v.endswith('.mp4')
        }
        # cache_video_meta_dict(video_meta_cache, video_meta_dict)
    else:
        print('Raw videos not found! Using cached metadata.')
        video_meta_dict = load_pickle(video_meta_cache)

    categories = diving48.load_categories()

    kwargs = {'meta_dict': video_meta_dict, 'emb_dir': emb_dir,
              'norm': norm, 'target_fps': target_fps}
    train_labels, train_actions = diving48.load_labels_and_embeddings(
        diving48.DIVING48_V1_TRAIN_FILE if use_v1
            else diving48.DIVING48_V2_TRAIN_FILE,
        **kwargs)
    train_embs = {k: v[1] for k, v in train_actions.items()}

    test_labels, test_actions = diving48.load_labels_and_embeddings(
        diving48.DIVING48_V1_TEST_FILE if use_v1
            else diving48.DIVING48_V2_TEST_FILE,
        **kwargs)
    test_embs = {k: v[1] for k, v in test_actions.items()}

    return categories, train_embs, train_labels, test_embs, test_labels


def load_tennis_data(dataset, emb_dir, norm):
    from action_dataset.load import (
        load_embs, load_actions, load_action_ids, to_categories)
    from action_dataset.eval import get_test_prefixes

    video_meta_cache = 'data/sports.cache/tennis.video_meta.pkl'
    if os.path.isdir(dataset_paths.TENNIS_VIDEO_DIR):
        video_meta_dict = {
            os.path.splitext(v)[0]: get_metadata(
                os.path.join(dataset_paths.TENNIS_VIDEO_DIR, v))
            for v in tqdm(os.listdir(dataset_paths.TENNIS_VIDEO_DIR),
                        desc='Loading video metadata')
            if v.endswith('.mp4')
        }
        # cache_video_meta_dict(video_meta_cache, video_meta_dict)
    else:
        print('Raw videos not found! Using cached metadata.')
        video_meta_dict = load_pickle(video_meta_cache)

    window_before, window_after = (0.5, 0.5)
    classes = [
        'forehand_topspin', 'forehand_slice', 'backhand_topspin',
        'backhand_slice', 'forehand_volley', 'backhand_volley', 'overhead'
    ]

    def parse_emb_video_name(v):
        player, clip_name = v.split('__', 1)
        video_name, start, end = clip_name.rsplit('_', 2)
        return (video_name, player, int(start), int(end), clip_name)

    emb_dict = {parse_emb_video_name(k): v for k, v in
                load_embs(emb_dir, norm).items()}

    actions = load_actions('action_dataset/{}/all.txt'.format(dataset))
    val_action_ids = load_action_ids(
        'action_dataset/{}/val.ids.txt'.format(dataset))
    test_prefixes = get_test_prefixes(dataset)
    print(test_prefixes)

    video_label_intervals = defaultdict(list)
    train_embs = {}
    train_labels = {}
    val_embs = {}
    val_labels = {}
    test_embs = {}
    test_labels = {}
    for action, label in actions.items():
        if label not in classes:
            continue

        label_idx = classes.index(label)
        base_video, player, frame = action.split(':')
        frame = int(frame)

        embs = None
        for v in emb_dict:
            if (
                    v[0] == base_video and v[1] == player and frame >= v[2]
                    and frame <= v[3]
            ):
                fps = video_meta_dict[v[-1]].fps
                mid_frame = frame - v[2]
                start_frame = max(0, int(mid_frame - fps * window_before))
                end_frame = int(mid_frame + fps * window_after)

                video_label_intervals[base_video + '_player'].append(
                    ((start_frame + v[2])/ fps, (end_frame + v[2]) / fps))

                action_embs = emb_dict[v][0][start_frame:end_frame]
                if len(action_embs) > 0:
                    embs = action_embs
                    break

        if base_video.startswith(test_prefixes):
            test_embs[action] = embs
            test_labels[action] = label_idx
        elif action in val_action_ids:
            val_embs[action] = embs
            val_labels[action] = label_idx
        else:
            train_embs[action] = embs
            train_labels[action] = label_idx
    return (to_categories(classes), train_embs, train_labels,
            val_embs, val_labels, test_embs, test_labels, video_label_intervals)


def load_fs_data(emb_dir, norm):
    from action_dataset.load import (
        load_embs, load_actions, load_action_ids, to_categories)
    from action_dataset.eval import get_test_prefixes

    video_meta_cache = 'data/sports.cache/fs.video_meta.pkl'
    if os.path.isdir(dataset_paths.FS_VIDEO_DIR):
        video_meta_dict = {
            os.path.splitext(v)[0]: get_metadata(
                os.path.join(dataset_paths.FS_VIDEO_DIR, v))
            for v in tqdm(os.listdir(dataset_paths.FS_VIDEO_DIR),
                        desc='Loading video metadata')
            if v.endswith('.mp4')
        }
        # cache_video_meta_dict(video_meta_cache, video_meta_dict)
    else:
        print('Raw videos not found! Using cached metadata.')
        video_meta_dict = load_pickle(video_meta_cache)

    window_before, window_after = (2.5, 0.5)
    classes = ['axel', 'lutz', 'flip', 'loop', 'salchow', 'toe_loop']

    emb_dict = load_embs(emb_dir, norm)
    actions = load_actions('action_dataset/fs/all.txt')
    val_action_ids = load_action_ids('action_dataset/fs/val.ids.txt')
    test_prefixes = get_test_prefixes('fs')

    video_label_intervals = defaultdict(list)
    train_embs = {}
    train_labels = {}
    val_embs = {}
    val_labels = {}
    test_embs = {}
    test_labels = {}
    for action, label in actions.items():
        if label not in classes:
            continue

        label_idx = classes.index(label)
        video, start_frame, end_frame = action.split(':')
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        fps = video_meta_dict[video].fps

        # Dilate
        mid_frame = (start_frame + end_frame) / 2
        start_frame = min(start_frame, int(mid_frame - fps * window_before))
        end_frame = max(end_frame, int(mid_frame + fps * window_after))
        embs = emb_dict[video][0][start_frame:end_frame]
        if len(embs) == 0:
            embs = None

        video_label_intervals[video].append(
            (start_frame / fps, end_frame / fps))

        if video.startswith(test_prefixes):
            test_embs[action] = embs
            test_labels[action] = label_idx
        elif action in val_action_ids:
            val_embs[action] = embs
            val_labels[action] = label_idx
        else:
            train_embs[action] = embs
            train_labels[action] = label_idx

    return (to_categories(classes), train_embs, train_labels,
            val_embs, val_labels, test_embs, test_labels,         video_label_intervals)


def run_action_recognition(
        categories, train_embs, train_labels,
        val_embs, val_labels, test_embs, test_labels,
        out_dir, algorithm, k, num_train_examples, few_shot_template,
        hidden_dim, attn, num_epochs, val_freq, n_trials,
        no_test_flip, load_weights
):
    def build_model(train_embs):
        if algorithm in SEQ_MODELS:
            assert k == 1
            model_kwargs = {
                'hidden_dim': hidden_dim,
                'num_epochs': num_epochs,
                'val_freq': val_freq,
                'early_term_val_num_epochs': num_epochs // 3
            }
            if algorithm in ['gru', 'lstm']:
                model_kwargs['use_attention'] = attn
            if val_embs is not None:
                model_kwargs['val_embs'] = val_embs
                model_kwargs['val_labels'] = val_labels
            if load_weights is not None:
                model_kwargs['load_weights'] = load_weights
            model = SeqModel(algorithm, train_embs, train_labels,
                             **model_kwargs)
        else:
            model = KnnModel(algorithm, train_embs, train_labels, k)
        return model


    def save_results(trial, ne, results, acc, model):
        print('Saving results:', out_dir)
        os.makedirs(out_dir, exist_ok=True)

        trial_str = 'trial{}_{}_{}'.format(
            trial, ne if ne > 0 else 'full', algorithm)
        save_confusion_matrix(
            [r[2] for r in results], [r[4] for r in results],
            os.path.join(
                out_dir, '{}.test_conf.norm_true.pdf'.format(trial_str)),
            norm='true')

        save_confusion_matrix(
            [r[2] for r in results], [r[4] for r in results],
            os.path.join(
                out_dir, '{}.test_conf.norm_pred.pdf'.format(trial_str)),
            norm='pred')

        test_pred_path = os.path.join(
            out_dir, '{}.test_pred.csv'.format(trial_str))
        with open(test_pred_path, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow([
                'sequence', 'actual', 'actual_name',
                'pred (acc={})'.format(acc), 'pred_name', 'neighbor'])
            for row in results:
                writer.writerow(row)

        if out_dir is not None and load_weights is None:
            model.save_model(os.path.join(
                out_dir, '{}.model.pt'.format(trial_str)))

    def run_trial(trial, train_embs, ne):
        print('Trial {}: train / val / test split: {} / {} / {}'.format(
              trial + 1, len(train_embs), len(val_embs), len(test_embs)))
        model = build_model(train_embs)

        errors = 0
        errors_untracked = 0
        results = []
        with tqdm(test_embs.items()) as pbar:
            for action_id, action_embs in pbar:
                pred, neighbor_id = model.predict(
                    action_embs, not no_test_flip)
                if pred < 0:
                    errors_untracked += 1

                actual = test_labels[action_id]
                if pred != actual:
                    errors += 1
                pred_name = categories[pred].name if pred in categories else ''
                results.append((action_id, actual, categories[actual].name,
                                pred, pred_name, neighbor_id))

                pbar.set_description('Predicting on test (err={:0.2f}%)'.format(
                                     errors / len(results) * 100))

        assert len(results) == len(test_embs)
        acc = 1 - errors / len(results)
        print('Accuracy:', acc)

        print(classification_report(
            [r[2] for r in results], [r[4] for r in results],
            labels=list(sorted({r[4] for r in results})), digits=3
        ))

        if out_dir is not None:
            save_results(trial, ne, results, acc, model)
        return acc

    for ne in num_train_examples:
        print('\nExperiment: {}-shot'.format(ne if ne > 0 else 'full'))

        all_accs = []
        for i in range(n_trials):
            if ne > 0:
                # if few_shot_template is None:
                #     exp_train_embs = sample_embeddings(
                #         train_embs, train_labels, ne)

                # Load premade train split
                exp_train_path = few_shot_template.format(ne, i)
                print('Loading:', exp_train_path)
                exp_train_ids = load_action_ids(exp_train_path)
                exp_train_embs = {a: b for a, b in train_embs.items()
                                  if a in exp_train_ids}
                print('  {} / {} = {:0.3f}'.format(
                    len(exp_train_embs), len(train_embs),
                    len(exp_train_embs) / len(train_embs) * 100))
            else:
                exp_train_embs = train_embs
            all_accs.append(run_trial(i, exp_train_embs, ne))

        print('Mean accuracy: {:0.3f} +/- {:0.3f}'.format(
              np.mean(all_accs) * 100, np.std(all_accs) * 100))


def run_action_retrieval(emb_dict, label_dict, hit_t, queryset=None):
    hit_t.sort()

    def get_embs(a):
        embs = emb_dict[a]
        if embs is not None and len(embs.shape) == 3:
            embs = embs.reshape((embs.shape[0], -1))
        return embs

    actions = list(sorted(emb_dict.keys()))

    neighbors = Neighbors(
        [get_embs(a) for a in actions], build_dtw_distance_fn())

    hit_counts = defaultdict(int)
    hit_precs = defaultdict(list)

    queries = enumerate(actions)
    if queryset is not None:
        print('Restricting query set to {}'.format(len(queryset)))
        queries = filter(lambda q: q[1] in queryset, queries)
    queries = list(queries)

    print()
    max_hit_thresh = max(hit_t) + 1
    for q_idx, q in tqdm(queries, desc='Retrieving'):
        hit_at = None
        hits = []
        embs_q = get_embs(q)
        if embs_q is not None:
            # +1 until we see the query
            idx_ofs = 1
            for j, (r_idx, _) in enumerate(
                    neighbors.find(embs_q, max_hit_thresh, 1)
            ):
                if r_idx == q_idx:
                    assert idx_ofs > 0
                    idx_ofs = 0
                    if j != 0:
                        print('Warning: query is not 0th result ({}, norm: {})'.format(q, np.linalg.norm(embs_q, 'fro')))
                else:
                    label_q = label_dict[q]
                    if label_q == label_dict[actions[r_idx]]:
                        if hit_at is None:
                            hit_at = j + idx_ofs
                        hits.append(j + idx_ofs)

        for h in hit_t:
            if hit_at is not None and h >= hit_at:
                hit_counts[h] += 1

            prec_at_h = 0
            if len(hits) > 0:
                prec_at_h = sum(int(x <= h) for x in hits) / h
            assert prec_at_h <= 1
            hit_precs[h].append(prec_at_h)

    def hit_rate(h):
        return hit_counts[h] / len(queries) * 100

    print(tabulate(
        [['%', *['{:0.2f}'.format(hit_rate(h)) for h in hit_t]]],
        headers=['hit@', *hit_t]))

    def hit_prec(h):
        return np.mean(hit_precs[h]).item() * 100

    print(tabulate(
        [['%', *['{:0.2f}'.format(hit_prec(h)) for h in hit_t]]],
        headers=['prec@', *hit_t]))


def print_dataset_stats(categories, labels, embs):
    counts = Counter([labels[x] for x in embs])
    lines = []
    for k, v in counts.most_common():
        lines.append((categories[k].name, v))
    lines.sort()
    lines.append(('Total', len(embs)))
    print(tabulate(lines, headers=['Class name', 'Count']))


def print_interval_span(interval_dict):
    def deoverlap(intervals):
        ret = []
        for a, b in sorted(intervals):
            if len(ret) == 0 or ret[-1][1] < a:
                ret.append((a, b))
            else:
                ret[-1] = (ret[-1][0], b)
        return ret

    total = 0
    for v in interval_dict.values():
        total += sum(b - a for a, b in deoverlap(v))
    print('Total label span: {} s'.format(total))


def print_average_interval(interval_dict):
    interval_lens = []
    for v, i in interval_dict.items():
        for a, b in i:
            interval_lens.append(b - a)
    print('Mean label len:', np.mean(interval_lens))


def main(emb_dir, dataset, out_dir, algorithm, num_train_examples,
         norm, k, hidden_dim, attn, target_fps, num_epochs,
         val_freq, n_trials, no_test_flip, retrieve, load_weights):

    intervals = None
    val_embs, val_labels = None, None
    if dataset.startswith('diving48'):
        (categories, train_embs, train_labels, test_embs, test_labels) = \
         load_diving48_data(emb_dir, norm, target_fps,
                            use_v1=dataset == 'diving48v1')
        few_shot_file = 'action_dataset/diving48/train_{}_{}.ids.txt'
        if num_epochs is None:
            if len(num_train_examples) > 1:
                num_epochs = DIVING48_LOW_SHOT_NUM_EPOCHS
            else:
                num_epochs = DIVING48_FULL_NUM_EPOCHS
    elif dataset == 'fx':
        (categories, train_embs, train_labels, test_embs, test_labels,
         intervals) = load_finegym_data(dataset, emb_dir, norm, target_fps)
        few_shot_file = 'action_dataset/finegym99/train_{}_{}.ids.txt'
        if num_epochs is None:
            num_epochs = DEFAULT_NUM_EPOCHS
    elif dataset.startswith('tennis'):
        (categories, train_embs, train_labels, val_embs, val_labels,
         test_embs, test_labels, intervals) = load_tennis_data(
             dataset, emb_dir, norm)
        few_shot_file = (
            'action_dataset/{}'.format(dataset) + '/train_{}_{}.ids.txt')
        if num_epochs is None:
            num_epochs = DEFAULT_NUM_EPOCHS
    elif dataset == 'fs':
        (categories, train_embs, train_labels, val_embs, val_labels,
         test_embs, test_labels, intervals) = load_fs_data(emb_dir, norm)
        few_shot_file = 'action_dataset/fs/train_{}_{}.ids.txt'
        if num_epochs is None:
            num_epochs = DEFAULT_NUM_EPOCHS

    if intervals:
        print_average_interval(intervals)
        print_interval_span(intervals)

    print('\nTrain dataset:')
    print_dataset_stats(categories, train_labels, train_embs)

    if val_embs is not None:
        print('\nVal dataset:')
        print_dataset_stats(categories, val_labels, val_embs)

    print('\nTest dataset:')
    print_dataset_stats(categories, test_labels, test_embs)

    if retrieve:
        train_embs.update(test_embs)
        train_labels.update(test_labels)
        if val_embs is not None:
            train_embs.update(val_embs)
            train_labels.update(val_labels)
        assert num_train_examples != [-1], \
            'Must specify -ne for retrieval thresholds. E.g., "-ne 1 10 25 50"'
        run_action_retrieval(
            train_embs, train_labels, num_train_examples,
            set(test_embs.keys()) if dataset == 'diving48' else None)
    else:
        if val_embs is None:
            # Report val accuracies on Diving48 and FineGym
            val_embs = test_embs
            val_labels = test_labels

        train_embs = {k: v for k, v in train_embs.items() if v is not None}
        run_action_recognition(
            categories, train_embs, train_labels,
            val_embs, val_labels, test_embs, test_labels,
            out_dir, algorithm, k, num_train_examples, few_shot_file,
            hidden_dim, attn, num_epochs, val_freq, n_trials, no_test_flip,
            load_weights)


if __name__ == '__main__':
    main(**vars(get_args()))

#!/usr/bin/env python3

import os
import argparse
import random
import math
from collections import Counter, defaultdict
from typing import NamedTuple
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba

import torch
torch.set_num_threads(8)

from util.io import store_json, load_json, load_text
from util.proposal import BaseProposalModel, EnsembleProposalModel
from util.video import get_metadata
from action_dataset.load import load_embs, load_actions
from action_dataset.eval import get_test_prefixes
import video_dataset_paths as dataset_paths


class DataConfig(NamedTuple):
    video_name_prefix: 'Optional[str]'
    classes: 'List[str]'
    window_before: float = 0.
    window_after: float = 0.


TENNIS_CLASSES = [
    'forehand_topspin', 'forehand_slice', 'backhand_topspin', 'backhand_slice',
    'forehand_volley', 'backhand_volley', 'overhead', 'serve', 'unknown_swing'
]
TENNIS_WINDOW = 0.1
TENNIS_MIN_SWINGS_FEW_SHOT = 5


DATA_CONFIGS = {
    'tennis': DataConfig(
        video_name_prefix=None,
        classes=TENNIS_CLASSES,
        window_before=TENNIS_WINDOW,
        window_after=TENNIS_WINDOW
    ),
    'tennis_front': DataConfig(
        video_name_prefix='front__',
        classes=TENNIS_CLASSES,
        window_before=TENNIS_WINDOW,
        window_after=TENNIS_WINDOW
    ),
    'tennis_back': DataConfig(
        video_name_prefix='back__',
        classes=TENNIS_CLASSES,
        window_before=TENNIS_WINDOW,
        window_after=TENNIS_WINDOW
    ),
    'fs_jump': DataConfig(
        video_name_prefix=None,
        classes=['axel', 'lutz', 'flip', 'loop', 'salchow', 'toe_loop'],
    ),
    'fx': DataConfig(video_name_prefix=None, classes=[])
}


class Label(NamedTuple):
    video: str
    value: str
    start_frame: int
    end_frame: int
    fps: float


EMB_FILE_SUFFIX = '.emb.pkl'
SEQ_MODELS = ['lstm', 'gru']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=list(DATA_CONFIGS.keys()))
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('--emb_dir', type=str)
    parser.add_argument('-nt', '--n_trials', type=int, default=1)
    parser.add_argument('--algorithm', type=str, choices=SEQ_MODELS,
                        default='gru')
    parser.add_argument('-ne', '--n_examples', type=int, default=-1)
    parser.add_argument('-tw', '--tennis_window', type=float)
    parser.add_argument('--_all', action='store_true')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int)
    return parser.parse_args()


def get_video_intervals(examples):
    result = defaultdict(list)
    for l in examples:
        result[l.video].append((l.start_frame, l.end_frame))

    def deoverlap(intervals):
        ret = []
        for a, b in sorted(intervals):
            if len(ret) == 0 or ret[-1][1] < a:
                ret.append((a, b))
            else:
                ret[-1] = (ret[-1][0], b)
        return tuple(ret)
    return {k: deoverlap(v) for k, v in result.items()}


class ProposalModel:

    MIN_TRAIN_EPOCHS = 25
    NUM_TRAIN_EPOCHS = 200

    def __init__(self, arch_type, emb_dict, train_labels, hidden_dim,
                 ensemble_size, splits=5, **kwargs):
        self.embs = emb_dict
        train_videos = list({l.video for l in train_labels
                             if l.video in emb_dict})

        def get_gt(video):
            vx, _ = emb_dict[video]
            vy = np.zeros(vx.shape[0], dtype=np.int32)
            for l in train_labels:
                if l.video == video:
                    vy[l.start_frame:l.end_frame] = 1
            return vx, vy

        X, y = [], []
        custom_split = None
        for i, v in enumerate(train_videos):
            vx, vy = get_gt(v)
            if len(vx.shape) == 3:
                if custom_split is None:
                    custom_split = []
                for j in range(vx.shape[1]):
                    X.append(vx[:, j, :])
                    y.append(vy)
                    custom_split.append(i)
            else:
                X.append(vx)
                y.append(vy)
        if custom_split is not None:
            assert len(custom_split) == len(X)

        if len(X) < ensemble_size:
            ensemble_size = splits = len(X)
            print('Too few videos for full ensemble:', ensemble_size)

        kwargs.update({
            'ensemble_size': ensemble_size, 'splits': splits,
            'num_epochs': ProposalModel.NUM_TRAIN_EPOCHS,
            'min_epochs': ProposalModel.MIN_TRAIN_EPOCHS,
            'custom_split': custom_split
        })

        if len(X) < ensemble_size:
            raise Exception('Not enough examples for ensemble!')
        else:
            self.model = EnsembleProposalModel(
                arch_type, X, y, hidden_dim, **kwargs)

    def predict(self, video):
        x = self.embs[video][0]
        if len(x.shape) == 3:
            return self.model.predict_n(
                *[x[:, i, :] for i in range(x.shape[1])])
        else:
            return self.model.predict(x)


LOC_TEMPORAL_IOUS = [0.1 * i for i in range(1, 10)]


@numba.jit(nopython=True)
def calc_iou(a1, a2, b1, b2):
    isect = min(a2, b2) - max(a1, b1)
    return isect / (max(a2, b2) - min(a1, b1)) if isect > 0 else 0


def compute_precision_recall_curve(is_tp, num_pos):
    recall = []
    precision = []
    tp, fp = 0, 0
    for p in is_tp:
        if p:
            tp += 1
        else:
            fp += 1
        recall.append(tp / num_pos)
        precision.append(tp / (tp + fp))
    return precision, recall


def compute_interpolated_precision(precision, recall):
    interp_recall = []
    interp_precision = []

    max_precision = 0
    min_recall = 1
    for i in range(1, len(recall) + 1):
        r = recall[-i]
        p = precision[-i]
        if r < min_recall:
            if len(interp_precision) == 0 or p > interp_precision[-1]:
                interp_recall.append(min_recall)
                interp_precision.append(max_precision)
        max_precision = max(max_precision, p)
        min_recall = min(min_recall, r)
    interp_recall.append(0)
    interp_precision.append(1)

    interp_precision.reverse()
    interp_recall.reverse()
    return interp_precision, interp_recall


def compute_ap(pc, rc):
    ipc, irc = compute_interpolated_precision(pc, rc)
    assert irc[0] == 0, irc[0]
    assert irc[-1] == 1, (irc[-1], len(irc))

    area = 0
    for i in range(len(irc) - 1):
        dr = irc[i + 1] - irc[i]
        assert dr > 0
        p = ipc[i + 1]
        if i > 0:
            assert p < ipc[i], (p, ipc[i])
        area += p * dr
    assert area >= 0 and area <= 1, area
    return area


def plot_proposal_dist(props):
    fig = plt.figure()
    plt.hist(x=[b - a for a, b in props], bins=50)
    plt.xlabel('num frames')
    plt.ylabel('frequency')
    plt.show()
    plt.close(fig)


def plot_precision_recall_curve(p, r):
    fig = plt.figure()
    plt.plot(r, p)
    ip, ir = compute_interpolated_precision(p, r)
    plt.step(ir, ip)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
    plt.close(fig)


def make_split(train_examples, is_tennis):
    print('Making a new split!')
    train_videos = list({l.video for l in train_examples})
    if is_tennis:
        # Split off the player
        train_videos = list({
            v.split('__', 1)[1] for v in train_videos})

    print('Videos:', len(train_videos))
    train_videos.sort()
    random.Random(42).shuffle(train_videos)
    if is_tennis:
        train_intervals = get_video_intervals(train_examples)

        def tennis_filter(t):
            front_video = 'front__' + t
            back_video = 'back__' + t
            # Dont sample videos with too few swings
            return (
                len(train_intervals.get(front_video, []))
                    >= TENNIS_MIN_SWINGS_FEW_SHOT
                and len(train_intervals.get(back_video, []))
                    >= TENNIS_MIN_SWINGS_FEW_SHOT)
        train_videos = list(filter(tennis_filter, train_videos))

    for v in train_videos:
        print(v)
    return train_videos


def run_localization(dataset_name, emb_dict, train_examples, test_examples,
                     n_examples, n_trials, algorithm, k, hidden_dim, batch_size,
                     out_dir, _all=False):
    test_video_ints = get_video_intervals(test_examples)
    test_video_int_count = sum(len(v) for v in test_video_ints.values())
    print('Test examples (non-overlapping):', test_video_int_count)

    mean_train_int_len = np.mean([
        t.end_frame - t.start_frame for t in train_examples])
    min_prop_len = 0.67 * math.ceil(mean_train_int_len)
    max_prop_len = 1.33 * math.ceil(mean_train_int_len)

    if n_examples == -1:
        exp_name = 'full train'
    else:
        exp_name = '{} shot'.format(n_examples)

    # Activation threshold ranges
    thresholds = (
        np.linspace(0.05, 0.5, 10) if 'tennis' in dataset_name
        else np.linspace(0.1, 0.9, 9))

    trial_results = []
    for trial in range(n_trials):
        if n_examples < 0:
            exp_train_examples = train_examples
        else:
            few_shot_file = \
                'action_dataset/{}/train.localize.{}.txt'.format(
                    'fs' if dataset_name.startswith('fs') else dataset_name,
                    trial)
            print('Loading split:', few_shot_file)
            train_videos = load_text(few_shot_file)
            train_videos = train_videos[:n_examples]
            exp_train_examples = [
                l for l in train_examples
                if (l.video in train_videos or
                    ('tennis' in dataset_name and
                        l.video.split('__', 1)[1] in train_videos))]

        kwargs = {}
        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        model = ProposalModel(algorithm, emb_dict, exp_train_examples,
                                hidden_dim, ensemble_size=k, **kwargs)
        results = []
        for video in tqdm(
                set(emb_dict) if _all else
                {l.video for l in test_examples if l.video in emb_dict},
                desc='Running {}'.format(exp_name)
        ):
            scores = model.predict(video)
            results.append((video, scores))

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, '{}_trial{}_{}_pred.json'.format(
                    'train{}'.format(len(exp_train_examples)
                                        if n < 0 else n),
                    trial, algorithm))
            store_json(out_path, {k: v.tolist() for k, v in results})

        def calc_ap_at_threshold(act_thresh):
            all_props = []
            for video, scores in results:
                props = BaseProposalModel.get_proposals(scores, act_thresh)
                for p, score in props:
                    all_props.append((video, p, score))
            all_props.sort(key=lambda x: -x[-1])
            # plot_proposal_dist([x[1] for x in all_props])

            aps_at_tiou = []
            for t_iou in LOC_TEMPORAL_IOUS:
                all_remaining = {}
                for video, gt_ints in test_video_ints.items():
                    all_remaining[video] = set(gt_ints)

                is_tp = []
                for video, p, score in all_props:

                    mid = (p[1] + p[0]) // 2
                    if p[1] - p[0] < min_prop_len:
                        p = (max(0, mid - min_prop_len // 2),
                                mid + min_prop_len // 2)
                    elif p[1] - p[0] > max_prop_len:
                        p = (max(0, mid - max_prop_len // 2),
                                mid + max_prop_len // 2)

                    # Only the first retrieval can be correct
                    video_remaining = all_remaining.get(video)
                    if video_remaining is None:
                        is_tp.append(False)
                    else:
                        recalled = []
                        for gt in video_remaining:
                            if calc_iou(*p, *gt) >= t_iou:
                                recalled.append(gt)

                        # Disallow subsequent recall of these ground truth
                        # intervals
                        for gt in recalled:
                            video_remaining.remove(gt)
                            if len(video_remaining) == 0:
                                del all_remaining[video]
                        is_tp.append(len(recalled) > 0)

                if len(is_tp) > 0 and any(is_tp):
                    pc, rc = compute_precision_recall_curve(
                        is_tp, test_video_int_count)

                    # if (
                    #         np.isclose(t_iou, 0.5)
                    #         and np.isclose(act_thresh, 0.2)
                    # ):
                    #     plot_precision_recall_curve(pc, rc)
                    aps_at_tiou.append(compute_ap(pc, rc))
                else:
                    aps_at_tiou.append(0)
            return aps_at_tiou

        all_aps = []
        for act_thresh in thresholds:
            all_aps.append(calc_ap_at_threshold(act_thresh))

        headers = ['tIoU', *['AP@{:0.2f}'.format(x) for x in thresholds]]
        rows = []
        for i, t_iou in enumerate(LOC_TEMPORAL_IOUS):
            rows.append([t_iou, *[x[i] for x in all_aps]])

        print(tabulate(rows, headers=headers))
        trial_results.append(np.array(all_aps))

    if len(trial_results) > 1:
        mean_result = trial_results[0] / n_trials
        for t in trial_results[1:]:
            mean_result += t / n_trials

        headers = ['tIoU', *['AP@{:0.2f}'.format(x) for x in thresholds]]
        rows = []
        for i, t_iou in enumerate(LOC_TEMPORAL_IOUS):
            rows.append(
                [t_iou, *[mean_result[j, i] for j in range(len(thresholds))]])
        print('\nMean across {} trials:'.format(len(trial_results)))
        print(tabulate(rows, headers=headers))


def load_tennis_data(config):
    def parse_video_name(v):
        v = os.path.splitext(v)[0]
        video_name, start, end = v.rsplit('_', 2)
        return (video_name, int(start), int(end), v)

    video_meta_dict = {
        parse_video_name(v): get_metadata(
            os.path.join(dataset_paths.TENNIS_VIDEO_DIR, v)
        ) for v in tqdm(os.listdir(dataset_paths.TENNIS_VIDEO_DIR),
                        desc='Loading video metadata')
        if v.endswith('.mp4')
    }

    actions = load_actions('action_dataset/tennis/all.txt')
    test_prefixes = get_test_prefixes('tennis')

    train_labels = []
    test_labels = []
    for action, label in actions.items():
        if label not in config.classes:
            continue

        base_video, player, frame = action.split(':')
        frame = int(frame)

        for k in video_meta_dict:
            if k[0] == base_video and k[1] <= frame and k[2] >= frame:
                fps = video_meta_dict[k].fps
                mid_frame = frame - k[1]
                start_frame = max(
                    0, int(mid_frame - fps * config.window_before))
                end_frame = int(mid_frame + fps * config.window_after)

                label = Label(
                    '{}__{}'.format(player, k[-1]),
                    'action', start_frame, end_frame, fps)
                break

        if base_video.startswith(test_prefixes):
            test_labels.append(label)
        else:
            train_labels.append(label)
    return train_labels, test_labels


def load_fs_data(config):
    video_meta_dict = {
        os.path.splitext(v)[0]: get_metadata(
            os.path.join(dataset_paths.FS_VIDEO_DIR, v))
        for v in tqdm(os.listdir(dataset_paths.FS_VIDEO_DIR),
                      desc='Loading video metadata')
        if v.endswith('.mp4')
    }

    actions = load_actions('action_dataset/fs/all.txt')
    test_prefixes = get_test_prefixes('fs')
    durations = []

    train_labels = []
    test_labels = []
    for action, label in actions.items():
        if label not in config.classes:
            continue

        video, start_frame, end_frame = action.split(':')
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        fps = video_meta_dict[video].fps

        # Dilate
        mid_frame = (start_frame + end_frame) / 2
        start_frame = min(
            start_frame, int(mid_frame - fps * config.window_before))
        end_frame = max(end_frame, int(mid_frame + fps * config.window_after))
        durations.append((end_frame - start_frame) / fps)
        label = Label(video, 'action', start_frame, end_frame, fps)

        if video.startswith(test_prefixes):
            test_labels.append(label)
        else:
            train_labels.append(label)
    print(np.mean(durations))
    return train_labels, test_labels


def load_fx_data(config):
    from finegym.util import ANNOTATION_FILE
    from sklearn.model_selection import train_test_split

    video_meta_dict = {
        os.path.splitext(v)[0]: get_metadata(
            os.path.join(dataset_paths.FX_VIDEO_DIR, v))
        for v in tqdm(os.listdir(dataset_paths.FX_VIDEO_DIR),
                      desc='Loading video metadata')
        if v.endswith('.mp4')
    }

    all_labels = []

    event_id = 2    # Female fx
    annotations = load_json(ANNOTATION_FILE)
    for video in annotations:
        for event, event_data in annotations[video].items():
            if event_data['event'] != event_id:
                continue

            video_name = '{}_{}'.format(video, event)
            if event_data['segments'] is None:
                print('{} has no segments'.format(video_name))
                continue

            for segment, segment_data in event_data['segments'].items():
                assert segment_data['stages'] == 1
                assert len(segment_data['timestamps']) == 1
                start, end = segment_data['timestamps'][0]
                fps = video_meta_dict[video_name].fps
                start_frame = int(max(0, fps * (start - config.window_before)))
                end_frame = int(fps * (end + config.window_after))

                all_labels.append(Label(
                    video_name, 'action', start_frame, end_frame, fps))

    _, test_videos = train_test_split(
        list(video_meta_dict.keys()), test_size=0.25)
    test_videos = set(test_videos)

    train_labels = []
    test_labels = []
    for l in all_labels:
        if l.video in test_videos:
            test_labels.append(l)
        else:
            train_labels.append(l)
    return train_labels, test_labels


def main(dataset, out_dir, n_trials, n_examples, tennis_window,
         emb_dir, _all, algorithm, norm, k, hidden_dim, batch_size):
    config = DATA_CONFIGS[dataset]
    emb_dict = load_embs(emb_dir, norm)

    def print_label_dist(labels):
        print('Videos:', len({l.video for l in labels}))
        for name, count in Counter(x.value for x in labels).most_common():
            print('  {} : {}'.format(name, count))

    if dataset.startswith('tennis'):
        if tennis_window is not None:
            config = config._replace(
                window_before=tennis_window,
                window_after=tennis_window)
        train_labels, test_labels = load_tennis_data(config)
    elif dataset.startswith('fs'):
        train_labels, test_labels = load_fs_data(config)
    else:
        train_labels, test_labels = load_fx_data(config)
    print('\nLoaded {} train labels'.format(len(train_labels)))
    print_label_dist(train_labels)

    print('\nLoaded {} test labels'.format(len(test_labels)))
    print_label_dist(test_labels)

    print('\nTrain / test split: {} / {}\n'.format(
          len(train_labels), len(test_labels)))

    run_localization(dataset, emb_dict, train_labels, test_labels,
                     n_examples, n_trials, algorithm, k, hidden_dim, batch_size,
                     out_dir, _all=_all)


if __name__ == '__main__':
    main(**vars(get_args()))

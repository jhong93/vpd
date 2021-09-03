import os
from typing import NamedTuple
import numpy as np

from util.io import load_json, load_pickle

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DIVING48_CATEGORY_FILE = os.path.join(DIR_PATH, 'data', 'Diving48_vocab.json')
DIVING48_V1_TRAIN_FILE = os.path.join(DIR_PATH, 'data', 'Diving48_train.json')
DIVING48_V1_TEST_FILE = os.path.join(DIR_PATH, 'data', 'Diving48_test.json')
DIVING48_V2_TRAIN_FILE = os.path.join(
    DIR_PATH, 'data', 'Diving48_V2_train.json')
DIVING48_V2_TEST_FILE = os.path.join(DIR_PATH, 'data', 'Diving48_V2_test.json')


class Category(NamedTuple):
    name: str
    stages: list


def load_categories():
    result = {}
    for i, seq in enumerate(load_json(DIVING48_CATEGORY_FILE)):
        result[i] = Category(' '.join(seq), seq)
    return result


def _normalize_rows(x):
    d = np.linalg.norm(x, axis=1, keepdims=True)
    d[d < 1e-12] = 1
    return x / d


def load_labels_and_embeddings(
        label_file, meta_dict=None, emb_dir=None, norm=False, target_fps=None
):
    labels = {}
    data = {}
    for action in load_json(label_file):
        video_id = action['vid_name']
        start_frame = action['start_frame']
        end_frame = action['end_frame']

        embs = []
        if emb_dir is not None:
            video_meta = meta_dict.get(video_id)

            sample_incr = 1
            if target_fps is not None:
                sample_incr = min(1, target_fps / video_meta.fps) + 0.01
            sample_balance = 0

            emb_path = os.path.join(emb_dir, video_id + '.emb.pkl')
            if os.path.isfile(emb_path):
                for frame_num, emb, _ in load_pickle(emb_path):
                    if frame_num >= start_frame and frame_num < end_frame:
                        if sample_balance >= 0:
                            sample_balance -= 1
                            embs.append(emb)
                        sample_balance += sample_incr

        if len(embs) > 0:
            embs = np.stack(embs)
            if np.isnan(embs).any():
                print('Found NaN')
                embs = np.nan_to_num(0, copy=False)
            if norm:
                embs = _normalize_rows(embs)
        else:
            embs = None
        labels[video_id] = action['label']
        data[video_id] = ((start_frame, end_frame), embs)
    return labels, data
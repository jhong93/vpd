import os
import math
from typing import NamedTuple
import numpy as np

from util.io import load_pickle

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

ANNOTATION_FILE = os.path.join(
    DIR_PATH, 'data', 'finegym_annotation_info_v1.1.json')
GYM99_CATEGORY_FILE = os.path.join(DIR_PATH, 'data', 'gym99_categories.txt')
GYM99_ABRV_CATEGORY_FILE = os.path.join(
    DIR_PATH, 'data', 'gym99_categories_abrv.txt')
GYM99_TRAIN_FILE = os.path.join(
    DIR_PATH, 'data', 'gym99_train_element_v1.1.txt')
GYM99_VAL_FILE = os.path.join(DIR_PATH, 'data', 'gym99_val_element.txt')


class Category(NamedTuple):
    class_id: int
    set_id: int
    g530_id: int
    event: str
    name: str


def _parse_label(s):
    return int(s.split(':', 1)[1].strip())


def load_categories(file_name):
    result = {}
    with open(file_name) as fp:
        for line in fp:
            clabel, slabel, glabel, data = line.split(';')
            clabel = _parse_label(clabel)
            slabel = _parse_label(slabel)
            glabel = _parse_label(glabel)
            event, name = data.strip()[1:].split(')', 1)
            result[clabel] = Category(clabel, slabel, glabel, event, name.strip())
    return result


def load_labels(file_name):
    result = {}
    with open(file_name) as fp:
        for line in fp:
            action_id, label = line.split(' ')
            result[action_id] = int(label)
    return result


def parse_full_action_id(s):
    s, action_id = s.split('_A_')
    video_id, event_id = s.split('_E_')
    return video_id, 'E_' + event_id, 'A_' + action_id


def _normalize_rows(x):
    d = np.linalg.norm(x, axis=1, keepdims=True)
    d[d < 1e-12] = 1
    return x / d


def load_actions(annotations, labels, meta_dict, emb_dir=None, norm=False,
                 pre_seconds=0, min_seconds=0, max_seconds=1000,
                 target_fps=None, interp_skipped=False):
    result = {}
    for full_action_id in labels:
        video_id, event_id, action_id = parse_full_action_id(full_action_id)
        video_event_id = '{}_{}'.format(video_id, event_id)

        video_meta = meta_dict.get(video_event_id)
        if video_meta is None:
            continue

        timestamps = annotations[video_id][event_id]['segments'][action_id]['timestamps']
        start, end = timestamps[0]
        if end - start > max_seconds:
            end = start + max_seconds
        elif end - start < min_seconds:
            end = start + min_seconds
        if pre_seconds > 0:
            start -= pre_seconds
        start = max(start, 0)

        start_frame = math.floor(start * video_meta.fps)
        end_frame = math.ceil(end * video_meta.fps)

        embs = []
        if emb_dir is not None:
            sample_incr = 1
            if target_fps is not None:
                sample_incr = min(1, target_fps / video_meta.fps)
            sample_balance = 1

            emb_path = os.path.join(emb_dir, video_event_id + '.emb.pkl')
            if os.path.isfile(emb_path):
                skipped_embs = []
                for frame_num, emb, _ in load_pickle(emb_path):
                    if frame_num >= start_frame and frame_num <= end_frame:
                        if sample_balance >= 0:
                            sample_balance -= 1

                            if interp_skipped and len(skipped_embs) > 0:
                                skipped_embs.append(emb)
                                emb = np.mean(skipped_embs, axis=0)
                                skipped_embs = []

                            embs.append(emb)
                        else:
                            if interp_skipped:
                                skipped_embs.append(emb)
                        sample_balance += sample_incr

        if len(embs) > 0:
            embs = np.stack(embs)
            if norm:
                embs = _normalize_rows(embs)
        else:
            embs = None
        result[full_action_id] = ((start_frame, end_frame), embs)
    return result

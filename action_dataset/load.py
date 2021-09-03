"""
Load embeddings for figure skating and tennis
"""

import os
import numpy as np
from typing import NamedTuple

from util.io import load_pickle


class Category(NamedTuple):
    name: str


def group_by_frame(embs):
    num_frames = max(x[0] for x in embs) + 1
    emb_shape = embs[0][1].shape
    if len(emb_shape) == 2:
        dense = np.zeros((num_frames, *emb_shape))
    else:
        dense = np.zeros((num_frames, emb_shape[-1]))
    counts = np.zeros(num_frames)
    for i, e, _ in embs:
        dense[i, :] += e
        counts[i] += 1

    frames = list(sorted({x[0] for x in embs}))
    for i in frames:
        if counts[i] > 0:
            dense[i, :] /= counts[i]

    # Interpolate
    prev_frame = frames[0]
    for frame in frames[1:]:
        gap = frame - prev_frame
        if gap > 1:
            for i in range(1, gap):
                a = i / gap
                dense[prev_frame + i, :] = (
                    a * dense[prev_frame, :] + (1. - a) * dense[frame, :])
        prev_frame = frame
    return (dense, counts > 0)


def normalize_rows(x):
    d = np.linalg.norm(x, axis=1 if len(x.shape) == 2 else 2, keepdims=True)
    d[d < 1e-12] = 1
    return x / d


def load_embs(emb_dir, norm, emb_ext='.emb.pkl'):
    print('Loading embs:', emb_dir)
    emb_dict = {
        emb_file[:-len(emb_ext)]: group_by_frame(
            load_pickle(os.path.join(emb_dir, emb_file))
        ) for emb_file in os.listdir(emb_dir)
        if emb_file.endswith(emb_ext)
    }
    if norm:
        emb_dict = {k: (normalize_rows(e), m)
                    for k, (e, m) in emb_dict.items()}
    print('  shape:', list(emb_dict.values())[0][0].shape[1:])
    return emb_dict


def load_actions(action_file):
    actions = {}
    with open(action_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            action, label = line.split(' ')
            actions[action] = label
    return actions


def load_action_ids(id_file):
    ids = set()
    with open(id_file) as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                ids.add(line)
    return ids


def to_categories(classes):
    return {i: Category(c) for i, c in enumerate(classes)}
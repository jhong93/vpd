#!/usr/bin/env python3

import os
import argparse
import numpy as np
from tqdm import tqdm

from util.io import store_pickle, load_pickle
from vpd_dataset.single_frame import _get_pose_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('emb_dir1', type=str)
    parser.add_argument('emb_dir2', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    return parser.parse_args()


def main(emb_dir1, emb_dir2, out_dir):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for emb_file in tqdm(os.listdir(emb_dir1)):
        embs1 = load_pickle(os.path.join(emb_dir1, emb_file))
        embs2 = load_pickle(os.path.join(emb_dir2, emb_file))
        assert len(embs1) == len(embs2)

        embs = []
        for a, b in zip(embs1, embs2):
            assert a[0] == b[0], 'Frame mismatch: {} != {} - {}'.format(
                a[0], b[0], emb_file)
            stacked = np.concatenate(
                (a[1], b[1]), axis=0 if len(a[1].shape) == 1 else 1)
            meta = a[2]
            key = 'kp_score'
            meta[key] = min(_get_pose_score(meta, 0.5),
                            _get_pose_score(b[2], 0.5))
            embs.append((a[0], stacked, meta))

        if out_dir is not None:
            store_pickle(os.path.join(out_dir, emb_file), embs)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
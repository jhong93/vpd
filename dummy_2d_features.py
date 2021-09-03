#!/usr/bin/env python3

"""
Convert COCO17 2D poses to dummy embeddings for 2D-VPD.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from util.io import store_pickle, load_gz_json
from vipe_dataset.dataset_base import normalize_2d_skeleton


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_dir', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('--no_flip', action='store_true')
    return parser.parse_args()


def main(pose_dir, out_dir, no_flip):
    for video_name in tqdm(sorted(os.listdir(pose_dir))):
        if video_name.endswith('.json.gz'):
            # Flat case
            video_pose_path = os.path.join(pose_dir, video_name)
            video_name = video_name.split('.json.gz')[0]
        else:
            # Nested case
            video_pose_path = os.path.join(
                pose_dir, video_name, 'coco_keypoints.json.gz')

        if not os.path.exists(video_pose_path):
            print('Not found:', video_pose_path)
            continue

        embs = []
        for frame_num, pose_data in load_gz_json(video_pose_path):
            raw_2d = np.array(pose_data[0][-1])
            pose_2d = normalize_2d_skeleton(raw_2d, False, to_tensor=False)
            emb = pose_2d[:, :2].flatten()  # drop confidence column
            meta = {'is_2d': True, 'kp_score': np.mean(pose_2d[:, 2] + 0.5).item()}
            if not no_flip:
                emb2 = normalize_2d_skeleton(
                    raw_2d, True, to_tensor=False)[:, :2].flatten()
                emb = np.stack([emb, emb2])
            embs.append((frame_num, emb, meta))

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            store_pickle(os.path.join(out_dir, video_name + '.emb.pkl'), embs)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
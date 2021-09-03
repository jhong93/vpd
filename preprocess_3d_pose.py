#!/usr/bin/env python3
"""
Prepare 3D pose data for view-invariant embedding by re-encoding 3D skeletons
as the hip coordinate, rotation, and parent-joint offsets.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from util.io import store_pickle, load_pickle
from vipe_dataset.people3d import load_3dpeople_skeleton
from vipe_dataset.human36m import load_human36m_skeleton
from vipe_dataset.nba2k import load_nba2k_skeleton
from vipe_dataset.amass import load_amass_skeleton

DATASETS = ['3dpeople', 'panoptic', 'human36m', 'nba2k', 'amass']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('-o', '--out_file', type=str)
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-vf', '--visualize_frequency', type=int, default=25)
    return parser.parse_args()


def process_3dpeople_data(data_dir, visualize, visualize_frequency):
    i = 0
    result = {}
    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        for action in tqdm(
                sorted(os.listdir(person_dir)),
                desc='Processing {}'.format(person)
        ):
            action_cam_dir = os.path.join(person_dir, action, 'camera01')

            frames = os.listdir(action_cam_dir)
            frame_pose3d = [None] * len(frames)
            for frame in frames:
                frame_path = os.path.join(action_cam_dir, frame)
                frame_no = int(os.path.splitext(frame)[0])
                frame_pose3d[frame_no - 1] = load_3dpeople_skeleton(
                    frame_path, visualize and i % visualize_frequency == 0)
                i += 1
            result[(person, action)] = frame_pose3d
    return result


def process_human36m_data(data_dir, visualize, visualize_frequency):
    import cdflib

    i = 0
    result = {}
    for person in os.listdir(data_dir):
        pose_dir = os.path.join(data_dir, person, 'MyPoseFeatures', 'D3_Positions')
        for action_file in tqdm(
                os.listdir(pose_dir), desc='Processing {}'.format(person)
        ):
            action = os.path.splitext(action_file)[0]
            action_path = os.path.join(pose_dir, action_file)
            cdf_data = cdflib.CDF(action_path)
            raw_poses = cdf_data.varget('Pose').squeeze()
            cdf_data.close()
            frame_poses = []
            for j in range(raw_poses.shape[0]):
                frame_poses.append(load_human36m_skeleton(
                    raw_poses[j, :],
                    visualize and i % visualize_frequency == 0))
                i += 1
            result[(person, action)] = frame_poses
    return result


def process_nba2k_data(data_dir, visualize, visualize_frequency):
    i = 0
    result = {}
    for person in os.listdir(data_dir):
        pose_file = os.path.join(
            data_dir, person, 'release_{}_2ku.pkl'.format(person))
        pose_data = load_pickle(pose_file)

        person_poses = []
        frames = os.listdir(os.path.join(data_dir, person, 'images', '2ku'))
        frames.sort()
        j3d = pose_data['j3d']
        assert len(frames) == len(j3d)
        for joints in tqdm(j3d, desc='Processing {}'.format(person)):
            person_poses.append(load_nba2k_skeleton(
                joints, visualize and i % visualize_frequency == 0))
            i += 1
        result[(person,)] = person_poses
    return result


def process_amass_data(data_dir, visualize, visualize_frequency):
    i = 0
    result = {}
    for seq in sorted(os.listdir(data_dir)):
        seq_dir = os.path.join(data_dir, seq)
        pose_file = os.path.join(data_dir, seq, 'pose.npy')
        if not os.path.isfile(pose_file):
            continue
        pose_arr = np.load(pose_file)

        frame_poses = []
        frames = list({
            f.split('_')[0] for f in os.listdir(seq_dir)
            if f.endswith(('jpg', 'png'))
        })
        frames.sort()
        assert len(frames) == pose_arr.shape[0], '{} has bad data'.format(seq)
        for j in tqdm(
                range(pose_arr.shape[0]), desc='Processing {}'.format(seq)
        ):
            frame_poses.append(
                load_amass_skeleton(
                    pose_arr[j, :, :],
                    visualize and i % visualize_frequency == 0))
            i += 1

        dataset, action = seq.split('_', 1)
        result[(dataset, action)] = frame_poses
    return result


def main(data_dir, dataset, out_file, visualize, visualize_frequency):
    if dataset == '3dpeople':
        pose3d = process_3dpeople_data(data_dir, visualize, visualize_frequency)
    elif dataset == 'human36m':
        pose3d = process_human36m_data(data_dir, visualize, visualize_frequency)
    elif dataset == 'nba2k':
        pose3d = process_nba2k_data(data_dir, visualize, visualize_frequency)
    elif dataset == 'amass':
        pose3d = process_amass_data(data_dir, visualize, visualize_frequency)
    else:
        raise NotImplementedError()

    if out_file is not None:
        store_pickle(out_file, pose3d)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

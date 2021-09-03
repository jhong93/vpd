#!/usr/bin/env python3

import os
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from models.module import FCResNet
from models.keypoint import Keypoint_EmbeddingModel
from vipe_dataset.dataset_base import (
    normalize_2d_skeleton, NUM_COCO_KEYPOINTS, NUM_COCO_BONES)
from util.io import load_gz_json, load_json, store_pickle


NUM_WORKERS = os.cpu_count() // 2
EMBED_BATCH_SIZE = 250


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_dir')
    parser.add_argument('model_dir')
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    parser.add_argument('-m', '--model_epoch', type=int)
    parser.add_argument('--allow_many_per_frame', action='store_true')
    parser.add_argument('--min_score', type=float, default=0)
    parser.add_argument('--no_flip', action='store_true',
                        help='Do not compute horizontally flipped embeddings')

    # For Diving48 and floor exercise
    parser.add_argument('--invert', action='store_true',
                        help='Compute embeddings on upside down poses')
    return parser.parse_args()


def mean_embs_by_frame(pred_embs, flip):
    grouped = defaultdict(list)
    for frame_num, emb, meta in pred_embs:
        grouped[frame_num].append((emb, meta))
    expected_shape = emb.shape

    def get_mean(emb_and_metas):
        embs, metas = zip(*emb_and_metas)
        if len(embs) == 1:
            emb, meta = embs[0], metas[0]
        else:
            emb = np.mean(embs, axis=0)
            meta = {'kp_score': min(m['kp_score'] for m in metas),
                    'is_mean': True}
        assert emb.shape == expected_shape
        return emb, meta

    result = []
    for frame_num, emb_and_metas in grouped.items():
        if flip:
            emb, mean_meta = get_mean(
                [x for x in emb_and_metas if not x[1]['is_flip']])
            emb_flip, _ = get_mean(
                [x for x in emb_and_metas if x[1]['is_flip']])
            mean_emb = np.stack((emb, emb_flip))
        else:
            mean_emb, mean_meta = get_mean(emb_and_metas)
        result.append((frame_num, mean_emb, mean_meta))
    result.sort(key=lambda x: x[0])
    return result


class VideoDataset(Dataset):

    def __init__(self, pose_dir, embed_bones, min_score, augment_flip, invert):
        super().__init__()

        videos = []
        for video_name in sorted(os.listdir(pose_dir)):
            if video_name.endswith('.json.gz'):
                # Flat case
                video_pose_path = os.path.join(pose_dir, video_name)
                video_name = video_name.split('.json.gz')[0]
            else:
                # Nested case
                video_pose_path = os.path.join(
                    pose_dir, video_name, 'coco_keypoints.json.gz')
            if os.path.exists(video_pose_path):
                videos.append((video_name, video_pose_path))

        self.videos = videos
        self.embed_bones = embed_bones
        self.min_score = min_score
        self.augment_flip = augment_flip
        self.invert = invert

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_name, video_pose_path = self.videos[idx]

        frames = []
        scores = []
        is_flip = []
        poses = []
        for frame_num, pose_data in load_gz_json(video_pose_path):
            for score, _, kp in pose_data:
                if score >= self.min_score:
                    kp = np.array(kp, dtype=np.float32)
                    if self.invert:
                        kp[:, 1] *= -1
                    kp_score = np.mean(kp[:, 2])

                    frames.append(frame_num),
                    scores.append(kp_score)
                    is_flip.append(False)
                    poses.append(normalize_2d_skeleton(
                        kp, False, include_bone_features=self.embed_bones))

                    if self.augment_flip:
                        frames.append(frame_num)
                        scores.append(kp_score)
                        is_flip.append(True)
                        poses.append(normalize_2d_skeleton(
                            kp, True, include_bone_features=self.embed_bones))

        return {'video': video_name, 'frame': np.array(frames),
                'score': np.array(scores), 'is_flip': np.array(is_flip),
                'pose': torch.stack(poses) if len(poses) > 0
                        else np.zeros(0)}


def load_embedding_model(model_dir, model_epoch=None):
    print('Loading embedding model:', model_dir)
    model_param_file = os.path.join(model_dir, 'config.json')
    model_params = load_json(model_param_file)

    embedding_dim = model_params['embedding_dim']
    encoder_arch = model_params['encoder_arch']
    embed_bones = model_params['embed_bones']

    print('Embedding dim:', embedding_dim)
    print('Encoder architecture:', encoder_arch)

    if model_epoch is None:
        model_name = 'best_epoch'
    else:
        model_name = 'epoch{:04d}'.format(model_epoch)
    print('Model name:', model_name)

    encoder_path = os.path.join(
        model_dir, '{}.encoder.pt'.format(model_name))
    print('Encoder weights:', encoder_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    encoder = FCResNet(
        (NUM_COCO_KEYPOINTS + NUM_COCO_BONES
         if embed_bones else NUM_COCO_KEYPOINTS) * 3,
        embedding_dim, *encoder_arch)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    return Keypoint_EmbeddingModel(encoder, {}, device), embed_bones


def main(pose_dir, model_dir, model_epoch, out_dir,
         allow_many_per_frame, min_score, no_flip, invert):
    model, embed_bones = load_embedding_model(model_dir, model_epoch)

    # Run inference
    dataset = VideoDataset(
        pose_dir, embed_bones, min_score, not no_flip, invert)
    loader = DataLoader(dataset, shuffle=False, num_workers=NUM_WORKERS)

    def write_embs(video_name, embs):
        if embs and video_name is not None and out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            if not allow_many_per_frame:
                embs = mean_embs_by_frame(embs, not no_flip)
            store_pickle(os.path.join(
                out_dir, '{}.emb.pkl'.format(video_name)
            ), embs)

    with tqdm(loader) as pbar:
        for video_data in pbar:
            video_name = video_data['video'][0]
            pbar.set_description(video_name)

            frames = video_data['frame'][0]
            if len(frames) > 0:
                embs = []
                scores = video_data['score'][0]
                is_flip = video_data['is_flip'][0]
                poses = video_data['pose'][0]

                for i in range(0, frames.shape[0], EMBED_BATCH_SIZE):
                    batch_embs = model.embed(poses[i:i + EMBED_BATCH_SIZE, :, :])
                    embs.extend((
                            frames[i + j].item(), batch_embs[j, :],
                            {'kp_score': scores[i + j].item(), 'is_mean': False,
                            'is_flip': is_flip[i + j].item()}
                        ) for j in range(batch_embs.shape[0]))
                write_embs(video_name, embs)

    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

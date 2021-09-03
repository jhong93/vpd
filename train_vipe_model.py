#!/usr/bin/env python3

import argparse
import os
import math
import re
import itertools
from typing import NamedTuple
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from util.io import store_json, load_json
from vipe_dataset import human36m, people3d, nba2k, amass
from vipe_dataset.util import render_3d_skeleton_views
from vipe_dataset.dataset_base import NUM_COCO_KEYPOINTS, NUM_COCO_BONES
from vipe_dataset.keypoint import (
    Human36MDataset, People3dDataset, Human36MDataset, NBA2kDataset,
    AmassDataset, Pairwise_People3dDataset)
from models.module import FCResNetPoseDecoder, FCPoseDecoder, FCResNet
from models.keypoint import Keypoint_EmbeddingModel

import vipe_dataset_paths as dataset_paths

NUM_RENDER_SEQS = 10

DATASETS_3D = ['3dpeople', 'human36m', 'nba2k', 'amass']
DATASETS_PAIR = ['3dpeople_pair']
DATASETS = DATASETS_3D + DATASETS_PAIR

LIFT_3D_WEIGHT = 1
USE_RESNET_DECODER = False
ENCODER_DROPOUT = 0.2
DECODER_DROPOUT = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--checkpoint_frequency', type=int, default=25)
    parser.add_argument('--render_preview_frequency', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--encoder_arch', type=int, nargs=2, default=(2, 1024),
                        help='Num blocks, hidden size')
    parser.add_argument('--decoder_arch', type=int, nargs=2, default=(2, 512),
                        help='Num blocks, hidden size'),
    parser.add_argument('--embed_bones', action='store_true')
    parser.add_argument('--model_select_contrast', action='store_true')
    parser.add_argument('--model_select_window', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no_camera_aug', action='store_true')
    return parser.parse_args()


def render_frame_sequences(model, dataset_name, dataset, count,
                           skeleton_decoder):
    count = min(count, dataset.num_sequences)
    for i in tqdm(range(count), desc='Render - {}'.format(dataset_name)):
        sequence = dataset.get_sequence(i)
        for data in sequence:
            part_norms = data['kp_offset_norms']

            # Normalize the longest part to 1
            part_norms = part_norms / np.max(part_norms)

            true3d = data['kp_offsets'] * part_norms[:, None]
            pred3d = model.predict3d(
                data['pose'], dataset_name
            ).reshape(true3d.shape[0], -1)[:, :3] * part_norms[:, None]
            render_im = render_3d_skeleton_views(
                [
                    skeleton_decoder(true3d),
                    skeleton_decoder(pred3d),
                ],
                labels=['true', 'pred'],
                title='[{}] person={}, action={}, camera={}'.format(
                    dataset_name, data['person'], data['action'],
                    data['camera'])
            )
            yield cv2.cvtColor(render_im, cv2.COLOR_RGB2BGR)


def save_video_preview(out_file, frames):
    vo = None
    for frame in frames:
        if vo is None:
            h, w, _ = frame.shape
            vo = cv2.VideoWriter(
                out_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        vo.write(frame)
    vo.release()
    print('Saved video:', out_file)


class PoseDataset(NamedTuple):
    name: str
    train: 'Dataset'
    val: 'Dataset'
    has_3d: bool = False
    skeleton_decoder: 'Function' = None
    pose_3d_shape: 'Tuple' = None
    mean_kp_offset_norms: 'List[float]' = None


def load_datasets(dataset_names, embed_bones, augment_camera):
    datasets = []
    if 'human36m' in dataset_names:
        train_dataset, val_dataset = Human36MDataset.load_default(
            dataset_paths.HUMAN36M_KEYPOINT_DIR,
            dataset_paths.HUMAN36M_3D_POSE_FILE, embed_bones, augment_camera)
        datasets.append(PoseDataset(
            'human36m', train_dataset, val_dataset,
            has_3d=True, skeleton_decoder=human36m.decode_skeleton_from_offsets,
            pose_3d_shape=train_dataset[0]['kp_features'].shape,
            mean_kp_offset_norms=train_dataset.mean_kp_offset_norms.tolist()))

    if '3dpeople' in dataset_names:
        train_dataset, val_dataset = People3dDataset.load_default(
            dataset_paths.PEOPLE_3D_KEYPOINT_DIR,
            dataset_paths.PEOPLE_3D_3D_POSE_FILE, embed_bones, augment_camera)
        datasets.append(PoseDataset(
            '3dpeople', train_dataset, val_dataset,
            has_3d=True, skeleton_decoder=people3d.decode_skeleton_from_offsets,
            pose_3d_shape=train_dataset[0]['kp_features'].shape,
            mean_kp_offset_norms=train_dataset.mean_kp_offset_norms.tolist()))

    if '3dpeople_pair' in dataset_names:
        train_dataset, val_dataset = Pairwise_People3dDataset.load_default(
            dataset_paths.PEOPLE_3D_KEYPOINT_DIR, 20, embed_bones)
        datasets.append(PoseDataset(
            '3dpeople_pair', train_dataset, val_dataset))

    if 'nba2k' in dataset_names:
        train_dataset, val_dataset = NBA2kDataset.load_default(
            dataset_paths.NBA2K_KEYPOINT_DIR, dataset_paths.NBA2K_3D_POSE_FILE,
            embed_bones)
        datasets.append(PoseDataset(
            'nba2k', train_dataset, val_dataset,
            has_3d=True, skeleton_decoder=nba2k.decode_skeleton_from_offsets,
            pose_3d_shape=train_dataset[0]['kp_features'].shape,
            mean_kp_offset_norms=train_dataset.mean_kp_offset_norms.tolist()))

    if 'amass' in dataset_names:
        train_dataset, val_dataset = AmassDataset.load_default(
            dataset_paths.AMASS_KEYPOINT_DIR, dataset_paths.AMASS_3D_POSE_FILE,
            embed_bones, augment_camera)
        datasets.append(PoseDataset(
            'amass', train_dataset, val_dataset,
            has_3d=True, skeleton_decoder=amass.decode_skeleton_from_offsets,
            pose_3d_shape=train_dataset[0]['kp_features'].shape,
            mean_kp_offset_norms=train_dataset.mean_kp_offset_norms.tolist()))

    return datasets


def get_model_params(encoder, decoders):
    params = list(encoder.parameters())
    for d in decoders.values():
        params.extend(d.parameters())
    return params


def save_model(save_dir, name, encoder, decoders, optimizer):
    torch.save(
        encoder.state_dict(),
        os.path.join(save_dir, '{}.encoder.pt'.format(name)))

    for k, v in decoders.items():
        torch.save(
            v.state_dict(),
            os.path.join(save_dir, '{}.decoder-{}.pt'.format(name, k)))

    torch.save(
        optimizer.state_dict(),
        os.path.join(save_dir, '{}.optimizer.pt'.format(name)))


def load_model(save_dir, name, encoder, decoders, optimizer, device):
    encoder_path = os.path.join(save_dir, '{}.encoder.pt'.format(name))
    print('Loading:', encoder_path)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    for k, decoder in decoders.items():
        decoder_path = os.path.join(
            save_dir, '{}.decoder-{}.pt'.format(name, k))
        print('Loading:', decoder_path)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    optimizer_path = os.path.join(save_dir, '{}.optimizer.pt'.format(name))
    print('Loading:', optimizer_path)
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))


def get_last_checkpoint(save_dir):
    last_epoch = -1
    for fname in os.listdir(save_dir):
        m = re.match(r'epoch(\d+).encoder.pt', fname)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(epoch, last_epoch)
    return last_epoch


def get_train_loaders(datasets, batch_size, num_load_workers):
    total = sum(len(d.train) for d in datasets)
    num_batches = math.ceil(total / batch_size)
    train_loaders = [
        (d.name, DataLoader(
            d.train, round(len(d.train) / num_batches),
            shuffle=True, num_workers=num_load_workers,
        )) for d in datasets
    ]
    print('Target # train batches:', num_batches)
    for dataset, loader in train_loaders:
        print('  {} has {} batches'.format(dataset, len(loader)))
        num_batches = len(loader)
    return train_loaders


def get_moving_avg_loss(losses, n, key):
    return np.mean([l[key] for l in losses[-n:]])


def main(
        num_epochs, batch_size, embedding_dim, encoder_arch,
        decoder_arch, embed_bones, dataset, save_dir, render_preview_frequency,
        checkpoint_frequency, model_select_contrast, model_select_window,
        learning_rate, resume, no_camera_aug
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    augment_camera = not no_camera_aug
    del no_camera_aug

    if resume:
        print('Resuming training from:', save_dir)
        assert os.path.exists(save_dir)
        old_config = load_json(os.path.join(save_dir, 'config.json'))
        num_epochs = old_config['num_epochs']
        batch_size = old_config['batch_size']
        learning_rate = old_config['learning_rate']
        embedding_dim = old_config['embedding_dim']
        encoder_arch = old_config['encoder_arch']
        decoder_arch = old_config['decoder_arch']
        embed_bones = old_config['embed_bones']
        augment_camera = old_config['augment_camera']
        dataset = [d['name'] for d in old_config['datasets']]
    else:
        assert dataset is not None
        if 'all' in dataset:
            print('Using all datasets!', DATASETS)
            dataset = DATASETS
        elif '3d' in dataset:
            print('Using 3d datasets!', DATASETS_3D)
            dataset = DATASETS_3D

    print('Device:', device)
    print('Num epochs:', num_epochs)
    print('Batch size:', batch_size)
    print('Embedding dim:', embedding_dim)
    print('Encoder arch:', encoder_arch)
    print('Decoder arch:', decoder_arch)
    print('Embed bones:', embed_bones)
    print('Augment camera:', augment_camera)

    datasets = load_datasets(dataset, embed_bones, augment_camera)
    n_train_examples = sum(len(d.train) for d in datasets)
    n_val_examples = sum(len(d.val) for d in datasets if d.val is not None)
    for vipe_dataset in datasets:
        print('Dataset:', vipe_dataset.name)
        print('', 'Train sequences:', len(vipe_dataset.train))
        if vipe_dataset.val is not None:
            print('', 'Val sequences:', len(vipe_dataset.val))

    num_load_workers = max(os.cpu_count(), 4)
    train_loaders = get_train_loaders(datasets, batch_size, num_load_workers)
    val_loaders = [
        (d.name, DataLoader(
            d.val, batch_size, shuffle=False, num_workers=num_load_workers
        )) for d in datasets if d.val is not None]

    encoder = FCResNet(
        (NUM_COCO_KEYPOINTS + NUM_COCO_BONES
         if embed_bones else NUM_COCO_KEYPOINTS) * 3,
        embedding_dim, *encoder_arch, dropout=ENCODER_DROPOUT)

    # Add a 3d pose decoder
    pose_decoder_targets = [(d.name, math.prod(d.pose_3d_shape))
                            for d in datasets if d.has_3d]

    if USE_RESNET_DECODER:
        # A bigger decoder is not always better
        decoders = {'3d': FCResNetPoseDecoder(
            embedding_dim, *decoder_arch, pose_decoder_targets,
            dropout=DECODER_DROPOUT)}
    else:
        decoders = {'3d': FCPoseDecoder(
            embedding_dim, [decoder_arch[1]] * decoder_arch[0],
            pose_decoder_targets, dropout=DECODER_DROPOUT)}

    # Wrapper that moves the models to the device
    model = Keypoint_EmbeddingModel(encoder, decoders, device)

    def get_optimizer():
        return optim.AdamW(get_model_params(encoder, decoders),
                           lr=learning_rate)

    optimizer = get_optimizer()
    scaler = GradScaler() if device == 'cuda' else None

    # Initialize the model
    if resume:
        last_checkpoint = get_last_checkpoint(save_dir)
        load_model(save_dir, 'epoch{:04d}'.format(last_checkpoint),
                   encoder, decoders, optimizer, device)
        start_epoch = last_checkpoint + 1
    else:
        start_epoch = 1

        # Save the model settings
        os.makedirs(save_dir)
        store_json(os.path.join(save_dir, 'config.json'), {
            'datasets': [{
                'name': d.name,
                '3d_pose_shape': d.pose_3d_shape,
                'mean_kp_offset_norms': d.mean_kp_offset_norms
            } for d in datasets],
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'embedding_dim': embedding_dim,
            'encoder_arch': encoder_arch,
            'decoder_arch': decoder_arch,
            'embed_bones': embed_bones,
            'augment_camera': augment_camera
        })

    # Initialize the loss history
    loss_file = os.path.join(save_dir, 'loss.json')
    if resume:
        losses = [x for x in load_json(loss_file) if x['epoch'] < start_epoch]
        best_val_loss = min(get_moving_avg_loss(
            losses[:i], model_select_window, 'val'
        ) for i in range(model_select_window, len(losses)))
        print('Resumed val loss:', best_val_loss)
    else:
        losses = []
        best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs + 1):
        with tqdm(
                desc='Epoch {} - train'.format(epoch), total=n_train_examples
        ) as pbar:
            train_contra_loss, train_loss, dataset_train_losses = model.epoch(
                train_loaders, optimizer=optimizer, scaler=scaler,
                progress_cb=lambda n: pbar.update(n), weight_3d=LIFT_3D_WEIGHT)

        dataset_val_losses = []
        with tqdm(
                desc='Epoch {} - val'.format(epoch), total=n_val_examples
        ) as pbar:
            val_contra_loss, val_loss, dataset_val_losses = model.epoch(
                val_loaders, progress_cb=lambda n: pbar.update(n),
                weight_3d=LIFT_3D_WEIGHT)

        losses.append({
            'epoch': epoch,
            'train': train_contra_loss if model_select_contrast else train_loss,
            'val': val_contra_loss if model_select_contrast else val_loss,
            'dataset_train': [('contrast', train_contra_loss)]
                              + list(dataset_train_losses.items()),
            'dataset_val': [('contrast', val_contra_loss)]
                            + list(dataset_val_losses.items())
        })

        def print_loss(name, total, contra, mv_avg):
            print('Epoch {} - {} loss: {:0.5f}, contra: {:0.3f} [mv-avg: {:0.5f}]'.format(
                  epoch, name, total, contra, mv_avg))

        mv_avg_val_loss = get_moving_avg_loss(losses, model_select_window, 'val')
        print_loss('train', train_loss, train_contra_loss,
                   get_moving_avg_loss(losses, model_select_window, 'train'))
        print_loss('val', val_loss, val_contra_loss, mv_avg_val_loss)

        if loss_file is not None:
            store_json(loss_file, losses)

        if epoch % render_preview_frequency == 0 and save_dir is not None:
            save_video_preview(
                os.path.join(save_dir, 'epoch{:04d}.train.mp4'.format(epoch)),
                itertools.chain(*[
                    render_frame_sequences(
                        model, d.name, d.train, NUM_RENDER_SEQS,
                        d.skeleton_decoder
                    ) for d in datasets if d.has_3d]))

            save_video_preview(
                os.path.join(save_dir, 'epoch{:04d}.val.mp4'.format(epoch)),
                itertools.chain(*[
                    render_frame_sequences(
                        model, d.name, d.val, NUM_RENDER_SEQS,
                        d.skeleton_decoder
                    ) for d in datasets if d.has_3d and d.val is not None]))

        if save_dir is not None:
            if mv_avg_val_loss < best_val_loss:
                print('New best epoch!')
                save_model(save_dir, 'best_epoch', encoder, decoders,
                           optimizer)
            if epoch % checkpoint_frequency == 0:
                print('Saving checkpoint: {}'.format(epoch))
                save_model(save_dir, 'epoch{:04d}'.format(epoch), encoder,
                           decoders, optimizer)

        best_val_loss = min(mv_avg_val_loss, best_val_loss)

    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

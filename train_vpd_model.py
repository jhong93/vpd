#!/usr/bin/env python3

import argparse
import os
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from util.io import store_json
from vpd_dataset.single_frame import GenericDataset, TennisDataset, PennDataset
from vpd_dataset.common import RGB_MEAN_STD
from action_dataset.eval import get_test_prefixes

from models.rgb import RGBF_EmbeddingModel
from models.util import step

import video_dataset_paths as dataset_paths


DATASETS = ['tennis', 'fs', 'fx', 'penn', 'diving48']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--checkpoint_frequency', type=int)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--img_dim', type=int, default=128)
    parser.add_argument('--flow_img', type=str)
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--encoder_arch', type=str, default='resnet34')
    parser.add_argument('--model_select_window', type=int, default=5)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no_test_video', action='store_true')
    parser.add_argument('--min_pose_score', type=float)
    dataset_group = parser.add_mutually_exclusive_group()

    # Teacher embedding directory
    dataset_group.add_argument('--emb_dir', type=str)

    # Only for Penn dataset
    dataset_group.add_argument('--penn_dir', type=str)
    return parser.parse_args()


class ModelTrainer:
    """Class for training the encoder. Discarded after training"""

    def __init__(self, encoder, motion):
        super().__init__()
        device = encoder.device
        self.encoder = encoder.to(device)

        if motion:
            from models.module import FCNet
            self.fcn_time = FCNet(
                encoder.emb_dim, [128, 128], 2 * encoder.emb_dim,
                dropout=0).to(device)

    def epoch(self, data_loader, optimizer=None, scaler=None, progress_cb=None):
        device = self.encoder.device

        self.encoder.eval() if optimizer is None else self.encoder.train()
        if hasattr(self, 'fcn_time'):
            self.fcn_time.eval() if optimizer is None else self.fcn_time.train()

        epoch_emb_loss = 0.
        epoch_emb_n = 0
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch in data_loader:

                with nullcontext() if scaler is None else autocast():
                    img = batch['img'].to(device)
                    n = img.shape[0]

                    emb = self.encoder(img)
                    gt_emb = batch['emb'].to(device)
                    if hasattr(self, 'fcn_time'):
                        emb = self.fcn_time(emb)
                    emb_loss = F.mse_loss(emb, gt_emb, reduction='sum')
                    loss = emb_loss

                if optimizer is not None:
                    step(optimizer, scaler, loss)

                epoch_emb_loss += emb_loss.item()
                epoch_emb_n += n
                if progress_cb is not None:
                    progress_cb(n)

        return epoch_emb_loss / epoch_emb_n

    def get_optimizer(self, learning_rate):
        params = list(self.encoder.parameters())
        if hasattr(self, 'fcn_time'):
            params.extend(self.fcn_time.parameters())
        return torch.optim.AdamW(params, lr=learning_rate), \
            GradScaler() if self.encoder.device == 'cuda' else None

    def save_model(self, save_dir, name):
        torch.save(self.encoder.state_dict(),
                   os.path.join(save_dir, '{}.encoder.pt'.format(name)))
        if hasattr(self, 'fcn_time'):
            torch.save(self.fcn_time.state_dict(),
                       os.path.join(save_dir, '{}.decoder.pt'.format(name)))


def get_moving_avg_loss(losses, n, key):
    return np.mean([l[key] for l in losses[-n:]])


def load_dataset(
        dataset, dataset_kwargs, emb_dir, penn_dir, no_test_video
):
    if dataset == 'tennis':
        if emb_dir is None:
            emb_dir = os.path.join(dataset_paths.TENNIS_ROOT_DIR, 'embs')
        if no_test_video:
            dataset_kwargs['exclude_prefixes'] = get_test_prefixes(dataset)
        train_dataset, val_dataset, emb_dim = TennisDataset.load_default(
            emb_dir, dataset_paths.TENNIS_CROP_DIR, **dataset_kwargs)

    elif dataset == 'fs':
        if emb_dir is None:
            emb_dir = os.path.join(dataset_paths.FS_ROOT_DIR, 'embs')
        if no_test_video:
            dataset_kwargs['exclude_prefixes'] = get_test_prefixes(dataset)
        train_dataset, val_dataset, emb_dim = GenericDataset.load_default(
            emb_dir, dataset_paths.FS_CROP_DIR, **dataset_kwargs)

    elif dataset == 'fx':
        if emb_dir is None:
            emb_dir = os.path.join(dataset_paths.FX_ROOT_DIR, 'embs')
        if no_test_video:
            import finegym.util as fg_util
            fg_test_prefixes = tuple([
                l.split('_A_')[0] for l in fg_util.load_labels(
                    fg_util.GYM99_VAL_FILE)
            ])
            dataset_kwargs['exclude_prefixes'] = fg_test_prefixes
        train_dataset, val_dataset, emb_dim = GenericDataset.load_default(
            emb_dir, dataset_paths.FX_CROP_DIR, **dataset_kwargs)

    elif dataset == 'diving48':
        if no_test_video:
            import diving48.util as diving48_util
            dataset_kwargs['exclude_prefixes'] = tuple(
                diving48_util.load_labels_and_embeddings(
                    diving48_util.DIVING48_V2_TEST_FILE)[0].keys())
        if emb_dir is None:
            emb_dir = os.path.join(dataset_paths.DIVING48_ROOT_DIR, 'embs')
        train_dataset, val_dataset, emb_dim = GenericDataset.load_default(
            emb_dir, dataset_paths.DIVING48_CROP_DIR, **dataset_kwargs)

    elif dataset == 'penn':
        assert penn_dir is not None
        train_dataset, val_dataset, emb_dim = PennDataset.load_default(
            penn_dir, **dataset_kwargs)
    else:
        raise NotImplementedError()
    return train_dataset, val_dataset, emb_dim


def main(
        dataset, num_epochs, batch_size, learning_rate, img_dim, flow_img,
        motion, encoder_arch, save_dir, model_select_window,
        checkpoint_frequency, pretrained, emb_dir, penn_dir,
        no_test_video, min_pose_score
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rgb_mean_std = RGB_MEAN_STD['resnet' if pretrained else dataset]

    dataset_kwargs = {
        'img_dim': img_dim, 'flow_img_name': flow_img,
        'embed_time': motion, 'rgb_mean_std': rgb_mean_std,
        'target_len': 20000
    }
    if min_pose_score is not None:
        dataset_kwargs['min_pose_score'] = min_pose_score

    (
        train_dataset, val_dataset, emb_dim
    ) = load_dataset(dataset, dataset_kwargs, emb_dir, penn_dir, no_test_video)

    print('Device:', device)
    print('Num epochs:', num_epochs)
    print('Batch size:', batch_size)
    print('Image dim:', img_dim)
    print('Use flow:', flow_img is not None)
    print('Embed time:', motion)
    print('Encoder arch:', encoder_arch)
    print('Dataset:')
    print('', 'Train images:', len(train_dataset))
    print('', 'Val images:', len(val_dataset))
    print('', 'Embedding dim:', emb_dim)
    print('', 'Min pose score:', min_pose_score)

    num_load_workers = min(os.cpu_count(), 8)
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_load_workers,
        persistent_workers=False)
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size, num_workers=num_load_workers,
            persistent_workers=False)

    encoder = RGBF_EmbeddingModel(encoder_arch, emb_dim, flow_img is not None,
                                  device, pretrained=pretrained)
    trainer = ModelTrainer(encoder, motion)

    optimizer, scaler = trainer.get_optimizer(learning_rate)

    # Save the model settings
    os.makedirs(save_dir)
    store_json(os.path.join(save_dir, 'config.json'), {
        'num_epochs': num_epochs, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'img_dim': img_dim,
        'use_flow': flow_img is not None,
        'motion': motion, 'emb_dim': emb_dim,
        'encoder_arch': encoder_arch, 'rgb_mean_std': rgb_mean_std
    })

    # Initialize the loss history
    loss_file = os.path.join(save_dir, 'loss.json')
    losses = []
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        with tqdm(
                desc='Epoch {} - train'.format(epoch), total=len(train_dataset)
        ) as pbar:
            train_loss = trainer.epoch(
                train_loader, optimizer=optimizer, scaler=scaler,
                progress_cb=lambda n: pbar.update(n))

        val_loss = float('nan')
        if val_loader is not None:
            with tqdm(
                    desc='Epoch {} - val'.format(epoch), total=len(val_dataset)
            ) as pbar:
                val_loss = trainer.epoch(
                    val_loader, progress_cb=lambda n: pbar.update(n))

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss,
            'dataset_train': [(dataset, train_loss)],
            'dataset_val': [(dataset, val_loss)]
        })

        moving_avg_val_loss = get_moving_avg_loss(
            losses, model_select_window, 'val')
        print('Epoch {} - train loss: {:0.4f} [avg: {:0.4f}] val loss: {:0.4f} [avg: {:0.4f}]'.format(
            epoch, train_loss,
            get_moving_avg_loss(losses, model_select_window, 'train'),
            val_loss, moving_avg_val_loss))
        if loss_file is not None:
            store_json(loss_file, losses)

        if save_dir is not None:
            if moving_avg_val_loss < best_val_loss:
                print('New best epoch!')
                trainer.save_model(save_dir, 'best_epoch')
            if (
                    checkpoint_frequency is not None
                    and epoch % checkpoint_frequency == 0
            ):
                print('Saving checkpoint: {}'.format(epoch))
                trainer.save_model(save_dir, 'epoch{:04d}'.format(epoch))
        best_val_loss = min(moving_avg_val_loss, best_val_loss)

    if save_dir is not None:
        print('Saving last epoch: {}'.format(epoch))
        trainer.save_model(save_dir, 'epoch{:04d}'.format(epoch))
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

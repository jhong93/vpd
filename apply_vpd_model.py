#!/usr/bin/env python3

import os
import argparse
import re
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from util.io import store_pickle, load_json
from vpd_dataset.single_frame import FrameDataset
from models.rgb import RGBF_EmbeddingModel
import video_dataset_paths as dataset_paths

BATCH_SIZE = 500


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['tennis', 'fs', 'fx', 'diving48'])
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-m', '--model_epoch', type=int,
                        help='Specify an epooh. Otherwise use the best one.')

    parser.add_argument('--jitter', type=int,
                        help='Create additional jittered features.')
    parser.add_argument('--no_flip', action='store_true',
                        help='Do not embed horizontal flips')

    parser.add_argument('--flow_img', type=str)
    return parser.parse_args()


def get_tennis_dataset(dataset_kwargs):
    tasks = []
    videos = []
    for video_file in tqdm(
            os.listdir(dataset_paths.TENNIS_VIDEO_DIR), desc='Loading dataset'
    ):
        if not video_file.endswith('.mp4'):
            continue
        video_name = os.path.splitext(video_file)[0]
        src_video_name, start_frame, end_frame = video_name.rsplit('_', 2)
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        for player in ['front', 'back']:
            player_video_name = '{}__{}'.format(player, video_name)

            video_id = len(videos)
            videos.append(player_video_name)

            count = 0
            for frame_num in range(start_frame, end_frame + 1):
                img_path_prefix = os.path.join(
                    dataset_paths.TENNIS_CROP_DIR, src_video_name, player,
                    str(frame_num))
                if not os.path.isfile(img_path_prefix + '.png'):
                    continue
                tasks.append((
                    video_id, frame_num - start_frame, img_path_prefix))
                count += 1
            if count == 0:
                print('{} has no crops'.format(player_video_name))
    return videos, FrameDataset(tasks, **dataset_kwargs)


def get_dataset(crop_dir, dataset_kwargs):
    img_re = re.compile(r'^\d+\.png$')

    tasks = []
    videos = []
    for video_name in tqdm(os.listdir(crop_dir), desc='Loading dataset'):
        video_crop_dir = os.path.join(crop_dir, video_name)
        if not os.path.isdir(video_crop_dir):
            continue

        video_id = len(videos)
        videos.append(video_name)
        for img_file in os.listdir(video_crop_dir):
            if not img_re.match(img_file):
                continue
            frame_num = int(os.path.splitext(img_file)[0])
            tasks.append((
                video_id, frame_num,
                os.path.join(video_crop_dir, str(frame_num))
            ))
    return videos, FrameDataset(tasks, **dataset_kwargs)


def main(dataset, model_dir, out_dir, model_epoch, flow_img, jitter, no_flip):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_param_file = os.path.join(model_dir, 'config.json')
    model_params = load_json(model_param_file)
    emb_dim = model_params['emb_dim']
    encoder_arch = model_params['encoder_arch']
    img_dim = model_params['img_dim']
    use_flow = model_params['use_flow']
    if use_flow:
        assert flow_img is not None, 'No flow image name specified'
    embed_time = model_params['embed_time']
    rgb_mean_std = model_params['rgb_mean_std']

    print('Embedding dim:', emb_dim)
    print('Encoder architecture:', encoder_arch)
    print('Image dim:', img_dim)
    print('Use flow:', use_flow, '(name = {})'.format(flow_img))
    print('Embed time:', embed_time)
    print('Flip:', not no_flip)
    print('RGB mean & std:', rgb_mean_std)

    dataset_kwargs = {
        'img_dim': img_dim, 'flow_img_name': flow_img,
        'rgb_mean_std': rgb_mean_std, 'augment_flip': not no_flip
    }
    if jitter is not None:
        print('Augment: jitter {}'.format(jitter))
        dataset_kwargs['augment_jitter'] = jitter

    if dataset == 'tennis':
        videos, dataset = get_tennis_dataset(dataset_kwargs)
    elif dataset == 'fs':
        videos, dataset = get_dataset(
            dataset_paths.FS_CROP_DIR, dataset_kwargs)
    elif dataset == 'fx':
        videos, dataset = get_dataset(
            dataset_paths.FX_CROP_DIR, dataset_kwargs)
    elif dataset == 'diving48':
        videos, dataset = get_dataset(
            dataset_paths.DIVING48_CROP_DIR, dataset_kwargs)
    else:
        raise NotImplementedError()

    if model_epoch is None:
        model_name = 'best_epoch'
    else:
        model_name = 'epoch{:04d}'.format(model_epoch)
    print('Model name:', model_name)

    encoder_path = os.path.join(model_dir, '{}.encoder.pt'.format(model_name))
    encoder = RGBF_EmbeddingModel(encoder_arch, emb_dim, use_flow, device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)

    batch_size = BATCH_SIZE
    if jitter is not None:
        batch_size = batch_size // (jitter + 1)
    if no_flip:
        batch_size *= 2

    with tqdm(total=len(dataset), desc='Embedding frames') as pbar:
        all_embs = [list() for _ in videos]
        for batch in DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=os.cpu_count() // 2
        ):
            video_ids = batch['video'].tolist()
            frame_nums = batch['frame'].tolist()
            n_batch, k, w, h, d = batch['img'].shape
            batch_embs = encoder.embed(batch['img'].view(-1, w, h, d)).reshape(
                                       (n_batch, k, -1))
            for i in range(n_batch):
                all_embs[video_ids[i]].append((
                    frame_nums[i],
                    batch_embs[i, :, :] if k > 1 else batch_embs[i, 0, :],
                    {}
                ))
            pbar.update(n_batch)

        if out_dir is not None:
            for video_name, embs in zip(videos, all_embs):
                if len(embs) > 0:
                    embs.sort()
                    out_path = os.path.join(
                        out_dir, '{}.emb.pkl'.format(video_name))
                    os.makedirs(out_dir, exist_ok=True)
                    store_pickle(out_path, embs)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

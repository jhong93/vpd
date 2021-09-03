import os
import random
import math
import numpy as np
import cv2
cv2.setNumThreads(0)
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from util.io import load_pickle, load_json
from util.video import crop_frame

from .common import _BaseDataset, _TrainDataset, JITTER_KWARGS, EMB_FILE_SUFFIX


DEFAULT_MIN_POSE_SCORE = 0.5

RANDOM_MASK = True
RANDOM_MASK_PROB = 0.5
RANDOM_NOISE_SD = math.sqrt(0.05)


def _randbool():
    return random.getrandbits(1) > 0


def _normalize_rows(x):
    if len(x.shape) == 1:
        return x / np.linalg.norm(x)
    else:
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def _get_pose_score(meta_dict, default=None):
    score = meta_dict.get('dp_score')
    if score is not None:
        return score
    score = meta_dict.get('kp_score')
    if score is not None:
        return score
    if default is not None:
        return default
    raise NotImplementedError()


class TennisDataset(_TrainDataset):

    def __getitem__(self, idx):
        video_name, player, frame_num, emb, emb_meta = self._get()
        video_player_dir = os.path.join(self.img_dir, video_name, player)

        flip = False
        if len(emb.shape) == 2:
            flip = self.augment and _randbool()
            emb = emb[int(flip)]

        img = self._load_image(
            os.path.join(video_player_dir, '{}.png'.format(frame_num)))

        if RANDOM_MASK and random.random() <= RANDOM_MASK_PROB:
            mask_path = os.path.join(
                video_player_dir, '{}.mask.png'.format(frame_num))
            if os.path.exists(mask_path):
                mask = self._load_bg_mask(mask_path)
                noise = torch.randn(img.shape) * RANDOM_NOISE_SD
                noise[:, mask] = 0
                img += noise
                # img[:, mask] = 0

        if self.flow:
            flow = self._load_flow(os.path.join(
                video_player_dir, '{}.{}.png'.format(
                    frame_num, self.flow_img_name)))
            img = torch.cat((img, flow))

        if flip:
            # Flip horizontally and then invert the x flow values
            img = torch.flip(img, (2,))
            if self.flow:
                img[3, :, :] *= -1
        if self.augment:
            img = self._random_crop(img)
        return {'emb': torch.FloatTensor(emb), 'img': img}

    @staticmethod
    def load_default(emb_dir, img_dir, img_dim, embed_time,
                     target_len, rgb_mean_std, flow_img_name=None,
                     min_pose_score=None, normalize_target=False,
                     exclude_prefixes=None):
        videos = []
        emb_dim = None
        for emb_file in os.listdir(emb_dir):
            if not emb_file.endswith(EMB_FILE_SUFFIX):
                continue
            video_name = emb_file.split(EMB_FILE_SUFFIX)[0]
            if (
                    exclude_prefixes is not None
                    and video_name.startswith(exclude_prefixes)
            ):
                print('Excluded:', video_name)
                continue

            video_embs = load_pickle(os.path.join(emb_dir, emb_file))
            videos.append((video_name, video_embs))
            if emb_dim is None:
                emb_dim = video_embs[0][1].shape[-1]
            else:
                assert emb_dim == video_embs[0][1].shape[-1]

        def get_embs(videos):
            result = []
            for video_name, video_embs in videos:
                player, video_name = video_name.split('__', 1)
                video_name, start_frame, _ = video_name.rsplit('_', 2)

                for i in range(len(video_embs)):
                    frame_num, emb_target, emb_meta = video_embs[i]
                    if min_pose_score is None:
                        score_thresh = DEFAULT_MIN_POSE_SCORE
                    else:
                        score_thresh = min_pose_score

                    if _get_pose_score(emb_meta) < score_thresh:
                        continue
                    if normalize_target:
                        emb_target = _normalize_rows(emb_target)

                    if embed_time:
                        # Get previous emb
                        if i == 0 or video_embs[i - 1][0] != frame_num - 1:
                            continue

                        emb_prev = video_embs[i - 1][1]
                        if normalize_target:
                            emb_prev = _normalize_rows(emb_prev)

                        emb_target = np.concatenate(
                            [emb_target, emb_target - emb_prev],
                            axis=0 if len(emb_target.shape) == 1 else 1)
                    result.append((
                        video_name, player, int(start_frame) + frame_num,
                        emb_target, emb_meta
                    ))
            return result

        train_data, val_data = train_test_split(
            get_embs(videos), test_size=0.2)
        sort_key = lambda x: x[:3]
        train_data.sort(key=sort_key)
        val_data.sort(key=sort_key)

        train_dataset = TennisDataset(
            train_data, img_dir, img_dim, rgb_mean_std, target_len,
            flow_img_name=flow_img_name)
        val_dataset = TennisDataset(
            val_data, img_dir, img_dim, rgb_mean_std,
            int(target_len * 0.2), flow_img_name=flow_img_name)
        return train_dataset, val_dataset, emb_dim


class GenericDataset(_TrainDataset):

    def __getitem__(self, idx):
        video_name, frame_num, emb, _ = self._get()

        flip = False
        if len(emb.shape) == 2:
            flip = self.augment and _randbool()
            emb = emb[int(flip), :]

        img = self._load_image(os.path.join(
            self.img_dir, video_name, '{}.png'.format(frame_num)))

        if RANDOM_MASK and random.random() <= RANDOM_MASK_PROB:
            mask_path = os.path.join(
                self.img_dir, video_name, '{}.mask.png'.format(frame_num))
            if os.path.exists(mask_path):
                mask = self._load_bg_mask(mask_path)
                noise = torch.randn(img.shape) * RANDOM_NOISE_SD
                noise[:, mask] = 0
                img += noise
                # img[:, mask] = 0

        if self.flow:
            flow = self._load_flow(os.path.join(
                self.img_dir, video_name, '{}.{}.png'.format(
                    frame_num, self.flow_img_name)))
            img = torch.cat((img, flow))

        if flip:
            # Flip horizontally and then invert the x flow values
            img = torch.flip(img, (2,))
            if self.flow:
                img[3, :, :] *= -1
        if self.augment:
            img = self._random_crop(img)
        return {'emb': torch.FloatTensor(emb), 'img': img}

    @staticmethod
    def load_default(emb_dir, img_dir, img_dim, embed_time,
                     target_len, rgb_mean_std, flow_img_name=None,
                     min_pose_score=None, normalize_target=False,
                     exclude_prefixes=None):
        all_data = []
        emb_dim = None
        for emb_file in os.listdir(emb_dir):
            if not emb_file.endswith(EMB_FILE_SUFFIX):
                continue
            video_name = emb_file.split(EMB_FILE_SUFFIX)[0]
            if (
                    exclude_prefixes is not None
                    and video_name.startswith(exclude_prefixes)
            ):
                print('Excluded:', video_name)
                continue

            video_embs = load_pickle(os.path.join(emb_dir, emb_file))
            for i in range(len(video_embs)):
                frame_num, emb_target, emb_meta = video_embs[i]
                if emb_dim is not None:
                    assert emb_target.shape[-1] == emb_dim, \
                        'Inconsistent emb dims {} != {}'.format(
                            emb_target.shape[-1], emb_dim)
                else:
                    emb_dim = emb_target.shape[-1]

                if min_pose_score is None:
                    score_thresh = DEFAULT_MIN_POSE_SCORE
                else:
                    score_thresh = min_pose_score

                if _get_pose_score(emb_meta) < score_thresh:
                    continue

                if normalize_target:
                    emb_target = _normalize_rows(emb_target)

                if embed_time:
                    # Get previous emb
                    if i == 0 or video_embs[i - 1][0] != frame_num - 1:
                        continue

                    emb_prev = video_embs[i - 1][1]
                    if normalize_target:
                        emb_prev = _normalize_rows(emb_prev)

                    emb_target = np.concatenate(
                        [emb_target, emb_target - emb_prev],
                        axis=0 if len(emb_target.shape) == 1 else 1)
                all_data.append((
                    video_name, frame_num, emb_target, emb_meta))

        print('Videos:', len({x[0] for x in all_data}))
        train_data, val_data = train_test_split(all_data, test_size=0.2)
        train_data.sort()
        val_data.sort()

        train_dataset = GenericDataset(
            train_data, img_dir, img_dim, rgb_mean_std, target_len,
            flow_img_name=flow_img_name)
        val_dataset = GenericDataset(
            val_data, img_dir, img_dim, rgb_mean_std,
            int(target_len * 0.2), flow_img_name=flow_img_name)
        return train_dataset, val_dataset, emb_dim


class PennDatasetUtil:

    FRAME_DIR = '/mnt/hdd/penn-action/Penn_Action/frames'

    PAD_PX = 25
    PAD_FRAC = 0.1

    @staticmethod
    def load_crop(seq, frame_num, box, img_dim, flip=False):
        frame_path = os.path.join(
            PennDatasetUtil.FRAME_DIR, seq, '{:06d}.jpg'.format(frame_num + 1))
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        x, y, w, h = [int(z) for z in box]
        crop = crop_frame(
            x, y, x + w, y + h, frame, make_square=True,
            pad_px=PennDatasetUtil.PAD_PX, pad_frac=PennDatasetUtil.PAD_FRAC)
        if flip:
            crop = crop[:, ::-1, :].copy()
        return cv2.resize(crop, (img_dim, img_dim))


class PennDataset(_TrainDataset):

    def __init__(self, data, img_dim, rgb_mean_std, target_len, **kwargs):
        super().__init__(
            data, None, img_dim, rgb_mean_std, target_len, **kwargs)

    def __getitem__(self, idx):
        seq_name, frame_num, is_flip, emb, box = self._get()
        img = PennDatasetUtil.load_crop(
            seq_name, frame_num, box, self.img_dim, flip=is_flip)
        img = self._transform(img)

        if self.flow:
            raise NotImplementedError()

        if self.augment:
            img = self._random_crop(img)
        return {'emb': emb, 'img': img}

    @staticmethod
    def load_default(penn_dir, img_dim, embed_time, rgb_mean_std,
                     target_len, flow_img_name=None,
                     min_pose_score=DEFAULT_MIN_POSE_SCORE):
        emb_dict = load_pickle(os.path.join(penn_dir, 'pose_embs.pkl'))
        box_dict = load_json(os.path.join(penn_dir, 'boxes.json'))
        emb_dim = None

        all_data = []
        for seq, embs in emb_dict.items():
            boxes = box_dict[seq]
            for i in range(len(embs)):
                frame_num, score, emb_target = embs[i]
                if emb_dim is None:
                    emb_dim = emb_target.shape[-1]
                else:
                    assert emb_dim == emb_target.shape[-1]

                if score < min_pose_score:
                    continue
                if embed_time:
                    if i == 0 or embs[i - 1][0] != frame_num - 1:
                        continue
                    prev_embs = embs[i - 1][2]
                    emb_target = np.concatenate(
                        [emb_target, emb_target - prev_embs],
                        axis=0 if len(emb_target.shape) == 1 else 1)
                all_data.append((
                    seq, frame_num, False, emb_target[0], boxes[frame_num]))
                all_data.append((
                    seq, frame_num, True, emb_target[1], boxes[frame_num]))

        train_data, val_data = train_test_split(all_data, test_size=0.2)
        train_data.sort()
        val_data.sort()

        train_dataset = PennDataset(
            train_data, img_dim, rgb_mean_std, target_len,
            flow_img_name=flow_img_name, augment=True)
        val_dataset = PennDataset(
            val_data, img_dim, rgb_mean_std, int(target_len * 0.2),
            flow_img_name=flow_img_name, augment=False)
        return train_dataset, val_dataset, emb_dim


class FrameDataset(_BaseDataset):

    def __init__(self, tasks, img_dim, rgb_mean_std,
                 augment_jitter=0, augment_flip=False, flow_img_name=None):
        super().__init__(
            img_dim, transforms.Normalize(*rgb_mean_std, inplace=True))
        self.tasks = tasks
        self.jitter_count = augment_jitter
        self.flip = augment_flip
        self.flow_img_name = flow_img_name
        self._jitter = transforms.ColorJitter(**JITTER_KWARGS)

    def __getitem__(self, idx):
        video, frame_num, img_path_prefix = self.tasks[idx]
        img = self._load_image('{}.png'.format(img_path_prefix))

        imgs = [img]
        for _ in range(self.jitter_count):
            imgs.append(self._jitter(img))

        flip_imgs = None
        if self.flip:
            flip_img = torch.flip(img, (2,))
            flip_imgs = [flip_img]
            for _ in range(self.jitter_count):
                imgs.append(self._jitter(flip_img))

        if self.flow_img_name is not None:
            flow = self._load_flow(
                '{}.{}.png'.format(img_path_prefix, self.flow_img_name))
            imgs = [torch.cat((x, flow)) for x in imgs]

            if flip_imgs:
                flip_flow = torch.flip(flow, (2,))
                flip_flow[0, :, :] *= -1
                flip_imgs = [torch.cat((x, flip_flow)) for x in flip_imgs]

        if flip_imgs:
            imgs += flip_imgs
        return {'video': video, 'frame': frame_num, 'img': torch.stack(imgs)}

    def __len__(self):
        return len(self.tasks)

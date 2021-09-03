import os
import random
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


EMB_FILE_SUFFIX = '.emb.pkl'

JITTER_KWARGS = {
    'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.05, 'hue': 0.05}

RGB_MEAN_STD = {
    'tennis': (
        (0.44157383614877077, 0.47029633580897046, 0.4534017568516162),
        (0.13526736314774856, 0.1208027074415591, 0.1261687563723076)
    ),
    'fs': (
        (0.5747710337842444, 0.5644043210903272, 0.6334494151377134),
        (0.21349823115367886, 0.21827191146692457, 0.20393919008463163)
    ),
    'fx': (
        (0.38402001736617936, 0.34764328219285123, 0.4099846773620623),
        (0.19505844565544309, 0.18984186888162677, 0.1989230425908947)
    ),
    'diving48': (
        (0.3411329922282787, 0.46349889258964044, 0.5162481674015696),
        (0.16302619019820488, 0.17092395707914718, 0.19266662199338647)
    ),
    'penn': (
        (0.43258389316320306, 0.4293850246457961, 0.383481774195889),
        (0.18936336742486998, 0.18502009571154798, 0.18244625387985822)
    ),
    'resnet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}


class _BaseDataset(torch.utils.data.Dataset):

    def __init__(self, img_dim, img_transform):
        super().__init__()
        self.img_dim = img_dim

        # Normalization
        self.__transform = img_transform

        # Random resized crop
        self.__randcrop = transforms.RandomResizedCrop(
            img_dim, scale=(0.5, 1.), ratio=(0.9, 1.1))

    def _load_image(self, img_path):
        assert os.path.isfile(img_path), '{} is not a file!'.format(img_path)
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        if h != self.img_dim or w != self.img_dim:
            rgb = cv2.resize(rgb, (self.img_dim, self.img_dim))
        return self.__transform(torch.FloatTensor(rgb).permute(2, 0, 1) / 255.)

    def _transform(self, rgb):
        return self.__transform(torch.FloatTensor(rgb).permute(2, 0, 1) / 255.)

    def _load_flow(self, flow_path):
        assert os.path.isfile(flow_path), '{} is not a file!'.format(flow_path)
        flow = cv2.imread(flow_path)
        h, w, _ = flow.shape
        if h != self.img_dim or w != self.img_dim:
            flow = cv2.resize(flow, (self.img_dim, self.img_dim))
        return torch.FloatTensor((flow[:, :, :2] / 255) - 0.5).permute(2, 0, 1)

    def _load_bg_mask(self, mask_path):
        assert os.path.isfile(mask_path), '{} is not a file!'.format(mask_path)
        mask = cv2.imread(mask_path)
        h, w, _ = mask.shape
        if h != self.img_dim or w != self.img_dim:
            mask = cv2.resize(mask, (self.img_dim, self.img_dim))
        return torch.squeeze(torch.BoolTensor(mask[:, :, 0] == 0))

    def _random_crop(self, img):
        return self.__randcrop(img)


class _TrainDataset(_BaseDataset):

    def __init__(self, data, img_dir, img_dim, rgb_mean_std,
                 target_len, augment=True, flow_img_name=None):
        norm_transform = transforms.Normalize(*rgb_mean_std, inplace=True)
        if augment:
            norm_transform = transforms.Compose([
                transforms.ColorJitter(**JITTER_KWARGS),
                norm_transform
            ])
        super().__init__(img_dim, norm_transform)
        self.data = data
        self.img_dir = img_dir
        self.target_len = target_len
        self.flow_img_name = flow_img_name
        self.augment = augment

    @property
    def flow(self):
        return self.flow_img_name is not None

    def __len__(self):
        return self.target_len

    def _get(self):
        return random.choice(self.data)

#!/usr/bin/env python3
"""
Modified version of demo.py from RAFT.

Copy to the RAFT directory to compute optical flow on crops.

Original: https://github.com/princeton-vl/RAFT
"""

import sys
sys.path.append('core')

import argparse
import os
import re
from multiprocessing.pool import ThreadPool
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm

from raft import RAFT


DEVICE = 'cuda'

PNG_COMPRESSION = [cv2.IMWRITE_PNG_COMPRESSION, 9]
OUT_SUFFIX = '.{}.png'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def get_paths(crop_dir):
    base_img_re = re.compile(r'^\d+.png$')

    results = []
    def helper(d):
        for f in os.listdir(d):
            if base_img_re.match(f):
                frame = f.split('.', 1)[0]
                results.append(os.path.join(d, frame))
            else:
                fp = os.path.join(d, f)
                if os.path.isdir(fp):
                    helper(fp)

    helper(crop_dir)
    return results


class CropDataset(Dataset):

    def __init__(self, crop_dir, overwrite, out_name):
        self.out_suffix = OUT_SUFFIX.format(out_name)
        paths = []
        for im_prefix in get_paths(crop_dir):
            flow_out_path = im_prefix + self.out_suffix
            if not overwrite and os.path.exists(flow_out_path):
                continue
            paths.append(im_prefix)
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        im_prefix = self.paths[idx]
        image1 = load_image(im_prefix + '.prev.png')
        image2 = load_image(im_prefix + '.png')
        return im_prefix + self.out_suffix, image1, image2


def to_img(flow, clip):
    flow = np.clip(flow, -clip, clip) + clip
    flow *= 255 / (2 * clip + 1)
    h, w, _ = flow.shape
    return np.dstack((flow.astype(np.uint8), np.full((h, w, 1), 128, np.uint8)))


def output_batch(out_paths, flow, clip, subtract_median):
    for i in range(len(out_paths)):
        fi = flow[i]
        if subtract_median:
            mf = np.median(fi, axis=(0, 1))
            fi -= mf
        cv2.imwrite(out_paths[i], to_img(fi, clip), PNG_COMPRESSION)


def demo(args):
    dataset = CropDataset(args.path, args.overwrite, args.out_name)
    loader = DataLoader(dataset, num_workers=os.cpu_count() // 2,
                        batch_size=args.batch_size)

    with ThreadPool() as io_workers:
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(DEVICE)
        model.eval()

        futures = []
        with tqdm(desc='Processing', total=len(dataset)) as pbar, torch.no_grad():
            for flow_out_paths, image1s, image2s in loader:
                flow_low, flow_up = model(
                    image1s.to(DEVICE), image2s.to(DEVICE),
                    iters=20, test_mode=True)

                flow_np = flow_up.permute(0, 2, 3, 1).cpu().numpy()
                futures.append(io_workers.apply_async(
                    output_batch,
                    (flow_out_paths, flow_np, args.clip, args.subtract_median)))
                pbar.update(flow_np.shape[0])

        for fut in tqdm(futures, desc='Writing'):
            fut.get()
        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="dataset for evaluation")
    parser.add_argument('--model', help="restore checkpoint",
                        default='models/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', type=bool, default=True,
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')

    parser.add_argument('--clip', type=int, default=20)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--subtract_median', action='store_true')
    parser.add_argument('--out_name', type=str, required=True,
                        help='Suffix for output. E.g. <frame>.<out_name>.png')
    args = parser.parse_args()

    demo(args)

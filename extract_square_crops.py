#!/usr/bin/env python3
"""
Extract square crops around the athletes

For each frame t, produces:
 - crop for t
 - crop for t - 1 (or k)
 - estimated mask for t
"""

import argparse
import os
from multiprocessing import Pool
import cv2
cv2.setNumThreads(0)
import numpy as np
from tqdm import tqdm

from util.io import load_json, load_gz_json, decode_png
from util.video import crop_frame

PAD_PX = 25
PAD_FRAC = 0.1
PNG_COMPRESSION = [cv2.IMWRITE_PNG_COMPRESSION, 9]

MASK_THRESHOLD = 0.8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_dir', type=str)
    parser.add_argument('video_dir', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-d', '--dim', type=int, default=128)
    parser.add_argument('--target_fps', type=int)
    parser.add_argument('--num_prev_frames', type=int, default=1)
    parser.add_argument('--no_smooth', action='store_true')
    return parser.parse_args()


class DelayBuffer:

    def __init__(self, n):
        self.buffer = [None] * n
        self.idx = 0

    def push(self, x):
        self.buffer[self.idx] = x
        self.idx = (self.idx + 1) % len(self.buffer)

    def get(self, i):
        return self.buffer[(self.idx - 1 - i) % len(self.buffer)]


def extract_crops(video_path, box_dict, mask_dict, out_dir, dim, target_fps,
                  num_prev_frames, smooth_boxes, visualize):

    vc = cv2.VideoCapture(video_path)
    n = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vc.get(cv2.CAP_PROP_FPS)

    prev_box = None
    prev_sample_gap = 1 if target_fps is None else round(fps / target_fps)
    buffer = DelayBuffer(num_prev_frames * (prev_sample_gap + 1))
    for frame_num in range(n):
        ret, frame = vc.read()
        assert ret

        box = box_dict.get(frame_num)
        if box is not None:
            x, y, w, h = box
            x2, y2 = x + w, y + h

            # Union the current box with the previous box
            if smooth_boxes and prev_box is not None:
                x, y = min(x, prev_box[0]), min(y, prev_box[1])
                x2 = max(x2, prev_box[0] + prev_box[2])
                y2 = max(y2, prev_box[1] + prev_box[3])

            crop_box = [int(x), int(y), int(x2), int(y2)]
            crop = crop_frame(
                *crop_box, frame, make_square=True, pad_px=PAD_PX,
                pad_frac=PAD_FRAC)

            # Apply the same crop to the mask
            mask_crop = None
            mask_data = [m for m in mask_dict.get(frame_num, [])
                         if m[0] > MASK_THRESHOLD]
            if len(mask_data) > 0:
                mask_data.sort()
                _, mask_box, raw_mask = mask_data[-1]
                mx, my, mw, mh = map(int, mask_box)
                mask_frame = np.zeros((*frame.shape[:2], 1), np.uint8)
                mask_frame[my:my + mh, mx:mx + mw, :][decode_png(raw_mask)] = 255
                mask_crop = crop_frame(
                    *crop_box, mask_frame, make_square=True, pad_px=PAD_PX,
                    pad_frac=PAD_FRAC)

            # Get prev crops
            prev_crops = []
            for i in range(1, num_prev_frames + 1):
                prev_frame = buffer.get(prev_sample_gap * i)
                if prev_frame is not None:
                    prev_crops.append(crop_frame(
                        *crop_box, prev_frame, make_square=True,
                        pad_px=PAD_PX, pad_frac=PAD_FRAC))
                else:
                    prev_crops.append(crop)

            if max(crop.shape[:2]) != dim:
                crop = cv2.resize(crop, (dim, dim))
                prev_crops = [cv2.resize(pc, (dim, dim)) for pc in prev_crops]
                if mask_crop is not None:
                    mask_crop = cv2.resize(mask_crop, (dim, dim))

            if visualize:
                cv2.imshow('person', np.hstack((crop, *prev_crops)))
                cv2.waitKey(100)

            if out_dir is not None:
                crop_path = os.path.join(out_dir, '{}.png'.format(frame_num))
                cv2.imwrite(crop_path, crop, PNG_COMPRESSION)

                for i, prev_crop in enumerate(prev_crops, 1):
                    prev_crop_path = os.path.join(
                        out_dir, '{}.prev{}.png'.format(
                            frame_num, i if i > 1 else ''))
                    cv2.imwrite(prev_crop_path, prev_crop, PNG_COMPRESSION)

                if mask_crop is not None:
                    mask_crop_path = os.path.join(
                        out_dir, '{}.mask.png'.format(frame_num))
                    cv2.imwrite(mask_crop_path, mask_crop, PNG_COMPRESSION)

        prev_box = box
        buffer.push(frame)
    vc.release()
    cv2.destroyAllWindows()


def extract_crops_for_video(
        video_name, boxes, video_dir, pose_dir, out_dir, dim, target_fps,
        num_prev_frames, smooth_crops, visualize
):
    video_path = os.path.join(video_dir, video_name + '.mp4')
    video_out_dir = None
    if out_dir is not None:
        video_out_dir = os.path.join(out_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
    box_dict = {a: b for a, b in boxes}
    mask_dict = dict(load_gz_json(
        os.path.join(pose_dir, video_name, 'mask.json.gz')))
    extract_crops(video_path, box_dict, mask_dict, video_out_dir, dim,
                  target_fps, num_prev_frames, smooth_crops, visualize)
    return video_name


def worker_func(args):
    return extract_crops_for_video(*args)


def main(pose_dir, video_dir, out_dir, dim, target_fps, num_prev_frames,
         no_smooth, visualize):
    video_names = [x for x in os.listdir(pose_dir)
                   if os.path.isdir(os.path.join(pose_dir, x))]

    box_dict = {v: load_json(os.path.join(pose_dir, v, 'boxes.json'))
                for v in video_names}
    worker_args = [
        (v, box_dict[v], video_dir, pose_dir, out_dir, dim, target_fps,
         num_prev_frames, not no_smooth, visualize)
        for v in video_names]

    if visualize:
        parallelism = 1
    else:
        parallelism = os.cpu_count() // 2
    with Pool(parallelism) as p, \
            tqdm(total=sum(len(v) for v in box_dict.values())) as pbar:
        for video_name in p.imap_unordered(worker_func, worker_args):
            pbar.set_description(video_name)
            pbar.update(len(box_dict[video_name]))
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

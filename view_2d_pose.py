#!/usr/bin/env python3

import os
import cv2
import argparse
import numpy as np
from PIL import Image, ImageDraw
from tqdm import trange

from util.io import load_gz_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')
    parser.add_argument('pose_file')
    parser.add_argument('-v', dest='vout_file')
    parser.add_argument('-vs', dest='vout_scale', type=float)
    return parser.parse_args()


coco_bones = (
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5),
    (4, 6), (5, 7)
)


def draw_keypoints(im, kp_poses, r=5, w=3, fill='white'):
    draw = ImageDraw.Draw(im)
    for _, _, kp in kp_poses:
        for a, b in coco_bones:
            x1, y1, _ = kp[a - 1]
            x2, y2, _ = kp[b - 1]
            draw.line((x1, y1, x2, y2), fill=fill, width=w)


def main(video_file, pose_file, vout_file, vout_scale):
    if os.path.isdir(pose_file):
        video_name = os.path.splitext(video_file.rsplit('/', 1)[-1])[0]
        pose_file = os.path.join(
            pose_file, video_name, 'coco_keypoints.json.gz')

    kp_dict = dict(load_gz_json(pose_file))

    vc = cv2.VideoCapture(video_file)
    fps = vc.get(cv2.CAP_PROP_FPS)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    exp_frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    if vout_scale == 1:
        vout_scale = None

    if vout_file is not None:
        if vout_scale is not None:
            vo_size = (int(width * vout_scale), int(height * vout_scale))
        else:
            vo_size = (width, height)
        vo = cv2.VideoWriter(
            vout_file, cv2.VideoWriter_fourcc(*'x264'), fps, vo_size)
    else:
        vo = None

    for frame_num in trange(exp_frame_count):
        ret, frame = vc.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        frame_kp = kp_dict.get(frame_num)
        if frame_kp is not None:
            draw_keypoints(im, frame_kp)

        frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        if vo is None:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if vout_scale is not None:
                frame = cv2.resize(frame, vo_size)
            vo.write(frame)

    vc.release()
    if vo is not None:
        vo.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(**vars(get_args()))

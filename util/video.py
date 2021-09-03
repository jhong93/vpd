import os
import cv2
import random
import numpy as np
from collections import namedtuple
from subprocess import check_call


VideoMetadata = namedtuple('VideoMetadata', [
    'fps', 'num_frames', 'width', 'height'
])


def _get_metadata(vc):
    fps = vc.get(cv2.CAP_PROP_FPS)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMetadata(fps, num_frames, width, height)


def get_metadata(video_path):
    vc = cv2.VideoCapture(video_path)
    try:
        return _get_metadata(vc)
    finally:
        vc.release()


def decode_frame(video_path, frame_num):
    vc = cv2.VideoCapture(video_path)
    try:
        meta = _get_metadata(vc)
        assert frame_num < meta.num_frames
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        is_ok, frame = vc.read()
        assert is_ok
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        vc.release()


def pick_frame(video_path):
    vc = cv2.VideoCapture(video_path)
    try:
        meta = _get_metadata(vc)
        return random.randint(0, meta.num_frames - 1)
    finally:
        vc.release()


def cut_segment(video_file, video_meta, out_file, start, end):
    print('Extracting:', out_file)
    s = start / video_meta.fps
    ms = int(s * 100) % 100
    s = int(s)
    cmd = [
        'ffmpeg', '-ss', '{}.{}'.format(s, ms), '-i', video_file,
        '-c:v', 'libx264', '-c:a', 'aac', '-frames:v', str(end - start),
        '-y', out_file
    ]
    check_call(cmd)


def cut_segment_cv2(video_file, video_meta, out_file, start, end):
    print('Extracting using cv2:', out_file)
    vc = cv2.VideoCapture(video_file)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = vc.get(cv2.CAP_PROP_FPS)

    vo = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MP4V'),
                         fps, (width, height))
    vc.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start):
        ret, frame = vc.read()
        assert ret
        vo.write(frame)

    vc.release()
    vo.release()


def cut_frames(video_file, video_meta, out_dir, start, end, width=640, height=360):
    print('Extracting:', out_dir)
    os.makedirs(out_dir)
    s = start / video_meta.fps
    ms = int(s * 100) % 100
    s = int(s)
    cmd = [
        'ffmpeg', '-ss', '{}.{}'.format(s, ms), '-i', video_file,
        '-frames:v', str(end - start), '-qscale:v', '2',
        '-vf', 'scale=w={w}:h={h}:force_original_aspect_ratio=1,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2'.format(w=width, h=height),
        '-y', os.path.join(out_dir, '%05d.jpg')
    ]
    check_call(cmd)
    return len(os.listdir(out_dir))


def crop_frame(x1, y1, x2, y2, frame, make_square=False,
               pad_px=None, pad_frac=None):
    if make_square:
        h, w = y2 - y1, x2 - x1
        if h > w:
            mx = (x1 + x2) // 2
            x1 = mx - h // 2
            x2 = mx + h // 2
            if x2 - x1 < h:
                x1 -= 1
            assert x2 - x1 == h, (x2 - x1, h)
        elif h < w:
            my = (y1 + y2) // 2
            y1 = my - w // 2
            y2 = my + w // 2
            if y2 - y1 < w:
                y1 -= 1
            assert y2 - y1 == w, (y2 - y1, w)
    h, w = y2 - y1, x2 - x1

    pad_x = pad_y = pad_px if pad_px is not None else 0
    if pad_frac is not None:
        pad_x = int(w * pad_frac)
        pad_y = int(h * pad_frac)
    if pad_x > 0:
        x1 -= pad_x
        x2 += pad_x
    if pad_y > 0:
        y1 -= pad_y
        y2 += pad_y

    crop = frame[max(y1, 0):y2, max(x1, 0):x2, :]
    fh, fw, _ = frame.shape
    px1 = -min(x1, 0)
    px2 = max(0, x2 - fw)
    py1 = -min(y1, 0)
    py2 = max(0, y2 - fh)
    crop = np.pad(crop, ((py1, py2), (px1, px2), (0, 0)),
                  mode='constant', constant_values=0)
    if make_square:
        assert crop.shape[0] == crop.shape[1], crop.shape
    return crop


def frames_to_video(out_file, frame_files, fps):
    vo = None
    if len(frame_files) > 0:
        for frame_file in frame_files:
            img = cv2.imread(frame_file)
            if vo is None:
                h, w, _ = img.shape
                vo = cv2.VideoWriter(
                    out_file, cv2.VideoWriter_fourcc(*'avc1'),
                    fps, (h, w))
            vo.write(img)
        vo.release()
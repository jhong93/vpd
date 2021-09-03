#!/usr/bin/env python3
"""
Cut figure skating videos by routines.
"""

import argparse
import os
import csv
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

from util.video import get_metadata, cut_segment

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--padding', type=int, default=0)
    return parser.parse_args()


def recut_single(video_file, segments, out_dir):
    video_meta = get_metadata(video_file)
    video_name = os.path.basename(video_file).rsplit('.')[0]
    for seq_num, (start, end) in enumerate(tqdm(segments, desc=video_name)):
        start_frame = int(start * video_meta.fps)
        end_frame = int((end + 1) * video_meta.fps)

        video_segment_name = '{}_{:02d}_{:08d}_{:08d}'.format(
            video_name, seq_num + 1, start_frame, end_frame)
        video_segment_file = os.path.join(
            out_dir, '{}.mp4'.format(video_segment_name))
        cut_segment(video_file, video_meta, video_segment_file,
                    start_frame, end_frame)


def parse_duration(s):
    hh, mm, ss = s.split(':')
    return (int(hh) * 60 + int(mm)) * 60 + int(ss)


def load_segments(segment_file):
    segment_dict = defaultdict(list)
    with open(segment_file) as fp:
        rd = csv.DictReader(fp)
        for row in rd:
            segment_dict[row['video']].append(
                (parse_duration(row['start']), parse_duration(row['end'])))
    return segment_dict


def main(video_dir, out_dir, padding):
    # Start/end frame annotations for routines
    segment_file = 'action_dataset/fs/segments.csv'

    segment_dict = load_segments(segment_file)

    worker_args = []
    for video_name, video_segments in segment_dict.items():
        video_file = os.path.join(video_dir, video_name + '.mkv')
        assert os.path.isfile(video_file), video_file

        worker_args.append((
            video_file,
            [(a - padding, b + padding) for a, b in video_segments],
            out_dir
        ))

    os.makedirs(out_dir, exist_ok=True)
    with Pool(8) as p:
        p.starmap(recut_single, worker_args)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

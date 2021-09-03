#!/usr/bin/env python3
"""
Cut the FineGym video into routines.
"""

import os
import math
import argparse
from tqdm import tqdm

from util.io import load_json
from util.video import get_metadata, cut_segment
from finegym.util import ANNOTATION_FILE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir')
    parser.add_argument('event')
    parser.add_argument('-o', '--out_dir')
    return parser.parse_args()


EVENT_TYPES = {
    'female_VT': 1,
    'female_FX': 2,
    'female_BB': 3,
    'female_UB': 4
}


def main(video_dir, event, out_dir):
    annotations = load_json(ANNOTATION_FILE)
    event_type_id = EVENT_TYPES[event]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for video, events in tqdm(annotations.items()):
        video_path = os.path.join(video_dir, '{}.mp4'.format(video))
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir, '{}.mkv'.format(video))

        video_meta = get_metadata(video_path)
        for event_id, event_data in events.items():
            timestamps = event_data['timestamps']
            assert len(timestamps) == 1, 'Too many timestamps for event'
            start, end = timestamps[0]
            start_frame = math.floor(start * video_meta.fps)
            end_frame = math.ceil(end * video_meta.fps)

            if event_data['event'] == event_type_id and out_dir:
                clip_out_path = os.path.join(
                    out_dir, '{}_{}.mp4'.format(video, event_id))
                if not os.path.exists(clip_out_path):
                    cut_segment(video_path, video_meta, clip_out_path,
                                start_frame, end_frame)
                else:
                    print('Already done:', clip_out_path)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))

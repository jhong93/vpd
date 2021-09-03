import json
import pickle
import base64
import gzip
from io import BytesIO
from PIL import Image
import numpy as np


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def store_json(fpath, obj):
    with open(fpath, 'w') as fp:
        json.dump(obj, fp)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, 'wt', encoding='ascii') as fp:
        json.dump(obj, fp)


def load_pickle(fpath):
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


def store_pickle(fpath, obj):
    with open(fpath, 'wb') as fp:
        pickle.dump(obj, fp)


def decode_png(data):
    if isinstance(data, str):
        data = base64.decodebytes(data.encode())
    else:
        assert isinstance(data, bytes)
    fstream = BytesIO(data)
    im = Image.open(fstream)
    return np.array(im)


def encode_png(data, optimize=True):
    im = Image.fromarray(data)
    fstream = BytesIO()
    im.save(fstream, format='png', optimize=optimize)
    s = base64.encodebytes(fstream.getvalue()).decode()
    return s


def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def store_text(fpath, s):
    with open(fpath, 'w') as fp:
        fp.write(s)


def parse_time(time_str):
    seconds = 0.
    tokens = time_str.split(':')
    assert len(tokens) <= 3
    for i, t in enumerate(tokens):
        seconds *= 60
        if i != len(tokens) - 1:
            seconds += int(t)
        else:
            seconds += float(t)
    return seconds

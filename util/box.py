from typing import NamedTuple
import cv2


class Box(NamedTuple):
    x: int
    y: int
    w: int
    h: int


def calc_iou(b1, b2):
    ix1, iy1 = max(b1.x, b2.x), max(b1.y, b2.y)
    ix2, iy2 = min(b1.x + b1.w, b2.x + b2.w), min(b1.y + b1.h, b2.y + b2.h)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    ia = iw * ih
    assert ia >= 0
    return ia / (b1.w * b1.h + b2.w * b2.h - ia)


def calc_union(b1, b2):
    x1 = min(b1.x, b2.x)
    y1 = min(b1.y, b2.y)
    x2 = max(b1.x + b1.w, b2.x + b2.w)
    y2 = max(b1.y + b1.h, b2.y + b2.h)
    return Box(x1, y1, x2 - x1, y2 - y1)


def calc_contains(box, x, y):
    return (x >= box.x and x <= box.x + box.w
            and y >= box.y and y <= box.y + box.h)
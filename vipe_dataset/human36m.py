from collections import namedtuple
import numpy as np
import cv2

from .util import render_points, get_canonical_orientation, flip_skeleton_offsets


_BaseSkeleton = namedtuple('_BaseSkeleton', [
    'hips',
    'spine',
    'neck',
    'nose',
    'head_top',
    'right_up_leg',
    'right_leg',
    'right_foot',
    'right_toe_base',
    'left_up_leg',
    'left_leg',
    'left_foot',
    'left_toe_base',
    'right_arm',
    'right_forearm',
    'right_hand',
    'right_wrist_end',
    'left_arm',
    'left_forearm',
    'left_hand',
    'left_wrist_end',
])

COLORS = tuple('bbbbbrrrrggggrrrrgggg')


class Skeleton(_BaseSkeleton):

    bones = (
        (0, 1), (1, 2), (2, 3), (2, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (2, 13), (13, 14), (14, 15), (15, 16),
        (2, 17), (17, 18), (18, 19), (19, 20),
    )

Z_UNIT = np.array([0., 0., 1.])

SCHEMA = """
0,Hips
1,RightUpLeg
2,RightLeg
3,RightFoot
4,RightToeBase
5,Site
6,LeftUpLeg
7,LeftLeg
8,LeftFoot
9,LeftToeBase
10,Site
11,Spine
12,Spine1   <--- ignore
13,Neck
14,Head
15,Site
16,LeftShoulder <--- shoulders are at the same point
17,LeftArm
18,LeftForeArm
19,LeftHand
20,LeftHandThumb <--- same as hand
21,Site
22,L_Wrist_End
23,Site
24,RightShoulder
25,RightArm
26,RightForeArm
27,RightHand
28,RightHandThumb
29,Site
30,R_Wrist_End
31,Site
"""

XFLIP_ROWS = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15]


def get_skeleton_parent_cossim(s):
    # Assume normed offsets
    def dot(i, j):
        return s[i, :].dot(s[j, :])
    return np.array([
        1, dot(0, 1), dot(1, 2), dot(2, 3),
        dot(2, 4), dot(4, 5), dot(5, 6), dot(6, 7),
        dot(2, 8), dot(8, 9), dot(9, 10), dot(10, 11),
        dot(0, 12), dot(12, 13), dot(13, 14), dot(14, 15),
        dot(0, 16), dot(16, 17), dot(17, 18), dot(18, 19)
    ])


EXTREMITY_ROWS = [7, 11, 15, 19]


def encode_skeleton_as_offsets(s):
    return np.stack([
        s.spine - s.hips,
        s.neck - s.spine,
        s.nose - s.neck,
        s.head_top - s.neck,
        s.left_arm - s.neck,
        s.left_forearm - s.left_arm,
        s.left_hand - s.left_forearm,
        s.left_wrist_end - s.left_hand,
        s.right_arm - s.neck,
        s.right_forearm - s.right_arm,
        s.right_hand - s.right_forearm,
        s.right_wrist_end - s.right_hand,
        s.left_up_leg - s.hips,
        s.left_leg - s.left_up_leg,
        s.left_foot - s.left_leg,
        s.left_toe_base - s.left_foot,
        s.right_up_leg - s.hips,
        s.right_leg - s.right_up_leg,
        s.right_foot - s.right_leg,
        s.right_toe_base - s.right_foot,
    ])


def decode_skeleton_from_offsets(offsets, as_ndarray=False):
    def get(i):
        return offsets[i, :]

    spine = get(0)
    neck = spine + get(1)
    nose = neck + get(2)
    head_top = neck + get(3)
    left_arm = neck + get(4)
    left_forearm = left_arm + get(5)
    left_hand = left_forearm + get(6)
    left_wrist_end = left_hand + get(7)
    right_arm = neck + get(8)
    right_forearm = right_arm + get(9)
    right_hand = right_forearm + get(10)
    right_wrist_end = right_hand + get(11)
    left_up_leg = get(12)
    left_leg = left_up_leg + get(13)
    left_foot = left_leg + get(14)
    left_toe_base = left_foot + get(15)
    right_up_leg = get(16)
    right_leg = right_up_leg + get(17)
    right_foot = right_leg + get(18)
    right_toe_base = right_foot + get(19)

    skeleton = Skeleton(
        hips=np.zeros(3), spine=spine, neck=neck, nose=nose, head_top=head_top,
        left_arm=left_arm, left_forearm=left_forearm, left_hand=left_hand,
        left_wrist_end=left_wrist_end,
        right_arm=right_arm, right_forearm=right_forearm, right_hand=right_hand,
        right_wrist_end=right_wrist_end,
        left_up_leg=left_up_leg, left_leg=left_leg, left_foot=left_foot,
        left_toe_base=left_toe_base, right_up_leg=right_up_leg,
        right_leg=right_leg, right_foot=right_foot,
        right_toe_base=right_toe_base,
    )
    return np.stack(skeleton[1:]) if as_ndarray else skeleton


def load_human36m_skeleton(pose, visualize):
    xyz = np.array(pose).reshape((-1, 3)).astype(np.float32) / 100
    assert xyz.shape == (32, 3)

    hips_raw = xyz[0, :]
    xyz -= hips_raw

    if visualize:
        img = render_points(xyz[:, 0], xyz[:, 2])
        cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    forward_vec = get_canonical_orientation(
        xyz[[0, 11, 12, 13, 17, 25], :],
        np.cross(xyz[17, :] - xyz[0, :], xyz[25, :] - xyz[0, :]), # left x Right
        xyz[13, :] - xyz[0, :]  # neck - hip
    )
    forward_vec[2] = 0 # drop z
    forward_vec /= np.linalg.norm(forward_vec)
    lateral_vec = np.cross(Z_UNIT, forward_vec)

    rot_mat = np.array([lateral_vec, forward_vec, Z_UNIT]).T
    xyz = xyz.dot(rot_mat)

    theta = np.degrees(np.arccos(lateral_vec[0]))
    if lateral_vec[1] < 0:
        theta = -theta

    if visualize:
        img = render_points(xyz[:, 0], xyz[:, 2])
        cv2.imshow('canonical', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = render_points(xyz[:, 1], xyz[:, 2])
        cv2.imshow('canonical_side', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def get(i):
        return xyz[i, :]

    s = Skeleton(
        hips=get(0), right_up_leg=get(1), right_leg=get(2), right_foot=get(3),
        right_toe_base=get(4), left_up_leg=get(6), left_leg=get(7),
        left_foot=get(8), left_toe_base=get(9), spine=get(12),
        neck=get(13), nose=get(14), head_top=get(15),
        left_arm=get(17),
        left_forearm=get(18), left_hand=get(19),
        left_wrist_end=get(22),
        right_arm=get(25),
        right_forearm=get(26), right_hand=get(27),
        right_wrist_end=get(30)
    )

    skeleton_offsets = encode_skeleton_as_offsets(s)

    if visualize:
        cv2.imshow('skeleton', cv2.cvtColor(
            render_points([x[0] for x in s], [x[2] for x in s],
                          c=COLORS, segs=s.bones), cv2.COLOR_RGB2BGR))
        cv2.imshow('skeleton_side', cv2.cvtColor(
            render_points([x[1] for x in s], [x[2] for x in s],
                          c=COLORS, segs=s.bones), cv2.COLOR_RGB2BGR))

        s2 = decode_skeleton_from_offsets(skeleton_offsets)
        cv2.imshow('reconstructed', cv2.cvtColor(
            render_points([x[0] for x in s2], [x[2] for x in s2],
                          c=COLORS, segs=s2.bones), cv2.COLOR_RGB2BGR))

        flipped_skeleton_offsets = flip_skeleton_offsets(
            skeleton_offsets, XFLIP_ROWS)
        s3 = decode_skeleton_from_offsets(flipped_skeleton_offsets)
        cv2.imshow('flipped', cv2.cvtColor(
            render_points([x[0] for x in s3], [x[2] for x in s3],
                          c=COLORS, segs=s3.bones), cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)

    return (hips_raw, theta, skeleton_offsets)

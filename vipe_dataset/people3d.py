from collections import namedtuple
import numpy as np
import cv2

from .util import render_points, get_canonical_orientation, flip_skeleton_offsets


_BaseSkeleton = namedtuple('_BaseSkeleton', [
    'hips',
    'spine',
    'spine1',
    'spine2',
    'neck',
    'head',
    'head_top',
    'right_eye',
    'left_eye',
    'left_shoulder',
    'left_arm',
    'left_forearm',
    'left_hand',
    'right_shoulder',
    'right_arm',
    'right_forearm',
    'right_hand',
    'left_up_leg',
    'left_leg',
    'left_foot',
    'left_toe_base',
    'right_up_leg',
    'right_leg',
    'right_foot',
    'right_toe_base',
])

COLORS = tuple('bbbbbbbrgggggrrrrggggrrrr')


class Skeleton(_BaseSkeleton):

    bones = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8),
        (4, 9), (9, 10), (10, 11), (11, 12), (4, 13), (13, 14),
        (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (0, 21),
        (21, 22), (22, 23), (23, 24)
    )


Z_UNIT = np.array([0., 0., 1.])

SCHEMA = """
1: Hips
2: Spine
3: Spine1
4: Spine2
5: Neck
6: Head
7: HeadTop_End <---- also wrong, flipped with RightEye
8: RightEye <---- label is wrong (sides are flipped)
9: LeftEye
10: LeftShoulder
11: LeftArm
12: LeftForeArm
13: LeftHand
14: LeftHandMiddle1
15: LeftHandMiddle2
16: LeftHandMiddle3
17: LeftHandMiddle4
18: LeftHandThumb1
19: LeftHandThumb2
20: LeftHandThumb3
21: LeftHandThumb4
22: LeftHandIndex1
23: LeftHandIndex2
24: LeftHandIndex3
25: LeftHandIndex4
26: LeftHandRing1
27: LeftHandRing2
28: LeftHandRing3
29: LeftHandRing4
30: LeftHandPinky1
31: LeftHandPinky2
32: LeftHandPinky3
33: LeftHandPinky4
34: RightShoulder
35: RightArm
36: RightForeArm
37: RightHand
38: RightHandMiddle1
39: RightHandMiddle2
40: RightHandMiddle3
41: RightHandMiddle4
42: RightHandThumb1
43: RightHandThumb2
44: RightHandThumb3
45: RightHandThumb4
46: RightHandIndex1
47: RightHandIndex2
48: RightHandIndex3
49: RightHandIndex4
50: RightHandRing1
51: RightHandRing2
52: RightHandRing3
53: RightHandRing4
54: RightHandPinky1
55: RightHandPinky2
56: RightHandPinky3
57: RightHandPinky4
58: RightUpLeg  <---- label is wrong (sides are flipped)
59: RightLeg
60: RightFoot
61: RightToeBase
62: RightToe_End
63: LeftUpLeg
64: LeftLeg
65: LeftFoot
66: LeftToeBase
67: LeftToe_End
"""

XFLIP_ROWS = [0, 1, 2, 3, 4, 5, 7, 6, 12, 13, 14, 15, 8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19]


def get_skeleton_parent_cossim(s):
    # Assume normed offsets
    def dot(i, j):
        return s[i, :].dot(s[j, :])
    return np.array([
        1, dot(0, 1), dot(1, 2), dot(2, 3), dot(3, 4),
        dot(4, 5), dot(4, 6), dot(4, 7),
        dot(3, 8), dot(8, 9), dot(9, 10), dot(10, 11),
        dot(3, 12), dot(12, 13), dot(13, 14), dot(14, 15),
        dot(0, 16), dot(16, 17), dot(17, 18), dot(18, 19),
        dot(0, 20), dot(20, 21), dot(21, 22), dot(22, 23)
    ])


EXTREMITY_ROWS = [5, 6, 7, 19, 23]


def encode_skeleton_as_offsets(s):
    return np.stack([
        s.spine - s.hips,
        s.spine1 - s.spine,
        s.spine2 - s.spine1,
        s.neck - s.spine2,
        s.head - s.neck,
        s.head_top - s.head,
        s.right_eye - s.head,
        s.left_eye - s.head,
        s.left_shoulder - s.neck,
        s.left_arm - s.left_shoulder,
        s.left_forearm - s.left_arm,
        s.left_hand - s.left_forearm,
        s.right_shoulder - s.neck,
        s.right_arm - s.right_shoulder,
        s.right_forearm - s.right_arm,
        s.right_hand - s.right_forearm,
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
    spine1 = spine + get(1)
    spine2 = spine1 + get(2)
    neck = spine2 + get(3)
    head = neck + get(4)
    head_top = head + get(5)
    right_eye = head + get(6)
    left_eye = head + get(7)
    left_shoulder = neck + get(8)
    left_arm = left_shoulder + get(9)
    left_forearm = left_arm + get(10)
    left_hand = left_forearm + get(11)
    right_shoulder = neck + get(12)
    right_arm = right_shoulder + get(13)
    right_forearm = right_arm + get(14)
    right_hand = right_forearm + get(15)
    left_up_leg = get(16)
    left_leg = left_up_leg + get(17)
    left_foot = left_leg + get(18)
    left_toe_base = left_foot + get(19)
    right_up_leg = get(20)
    right_leg = right_up_leg + get(21)
    right_foot = right_leg + get(22)
    right_toe_base = right_foot + get(23)
    skeleton = Skeleton(
        hips=np.zeros(3), spine=spine, spine1=spine1, spine2=spine2, neck=neck,
        head=head, head_top=head_top, right_eye=right_eye, left_eye=left_eye,
        left_shoulder=left_shoulder, left_arm=left_arm,
        left_forearm=left_forearm, left_hand=left_hand,
        right_shoulder=right_shoulder, right_arm=right_arm,
        right_forearm=right_forearm, right_hand=right_hand,
        left_up_leg=left_up_leg, left_leg=left_leg, left_foot=left_foot,
        left_toe_base=left_toe_base,
        right_up_leg=right_up_leg, right_leg=right_leg, right_foot=right_foot,
        right_toe_base=right_toe_base,
    )
    return np.stack(skeleton[1:]) if as_ndarray else skeleton


def load_3dpeople_skeleton(fpath, visualize):
    uvdxyz = np.loadtxt(fpath)
    assert uvdxyz.shape == (67, 6)

    xyz = uvdxyz[:, 3:]
    hips_raw = xyz[0, :]
    xyz -= hips_raw

    if visualize:
        img = render_points(xyz[:, 0], xyz[:, 2])
        cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Rotate to facing forward based on shoulders and hips
    forward_vec = get_canonical_orientation(
        xyz[[0, 1, 2, 3, 9, 33], :],
        np.cross(xyz[9, :] - xyz[0, :], xyz[33, :] - xyz[0, :]), # left x Right
        xyz[4, :] - xyz[0, :] # neck - hip
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
        return xyz[i - 1, :]

    s = Skeleton(
        hips=get(1), spine=get(2), spine1=get(3), spine2=get(4), neck=get(5),
        head=get(6), head_top=get(9), left_eye=get(8), right_eye=get(7),
        left_shoulder=get(10), left_arm=get(11), left_forearm=get(12),
        left_hand=get(13),
        right_shoulder=get(34), right_arm=get(35), right_forearm=get(36),
        right_hand=get(37),
        left_up_leg=get(58), left_leg=get(59), left_foot=get(60),
        left_toe_base=get(61),
        right_up_leg=get(63), right_leg=get(64), right_foot=get(65),
        right_toe_base=get(66)
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

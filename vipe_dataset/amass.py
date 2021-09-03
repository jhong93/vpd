from collections import namedtuple
import numpy as np
import cv2

from .util import render_points, get_canonical_orientation, flip_skeleton_offsets


_BaseSkeleton = namedtuple('_BaseSkeleton', [
    'spine1',
    'spine2',
    'spine3',
    'neck',
    'head',
    'head_top',

    'l_hip',
    'l_knee',
    'l_ankle',
    'l_foot',
    'r_hip',
    'r_knee',
    'r_ankle',
    'r_foot',

    'l_collar',
    'l_shoulder',
    'l_elbow',
    'l_wrist',
    'r_collar',
    'r_shoulder',
    'r_elbow',
    'r_wrist'
])

COLORS = tuple('bbbbbbggggrrrrggggrrrr')


class Skeleton(_BaseSkeleton):

    bones = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 7), (7, 8), (8, 9),
        (0, 10), (10, 11), (11, 12), (12, 13),
        (3, 14), (14, 15), (15, 16), (16, 17),
        (3, 18), (18, 19), (19, 20), (20, 21)
    )


Z_UNIT = np.array([0., 0., 1.])

SCHEMA = """
This is bogus...
joint_names = {
    0: 'L_Hip',
    3: 'L_Knee',
    6: 'L_Ankle',
    9: 'L_Foot',

    1: 'R_Hip',
    4: 'R_Knee',
    7: 'R_Ankle',
    10: 'R_Foot',

    2: 'Spine1',
    5: 'Spine2',
    8: 'Spine3',
    11: 'Neck',
    14: 'Head',

    12: 'L_Collar',
    15: 'L_Shoulder',
    17: 'L_Elbow',
    19: 'L_Wrist',
    13: 'R_Collar',
    16: 'R_Shoulder',
    18: 'R_Elbow',
    20: 'R_Wrist',
}
"""

XFLIP_ROWS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 17, 18, 19, 20, 13, 14, 15, 16]


def get_skeleton_parent_cossim(s):
    # Assume normed offsets
    def dot(i, j):
        return s[i, :].dot(s[j, :])
    return np.array([
        1, dot(0, 1), dot(1, 2), dot(2, 3), dot(3, 4),
        dot(0, 5), dot(5, 6), dot(6, 7), dot(7, 8),
        dot(0, 9), dot(9, 10), dot(10, 11), dot(11, 12),
        dot(2, 13), dot(13, 14), dot(14, 15), dot(15, 16),
        dot(2, 17), dot(17, 18), dot(18, 19), dot(19, 20)
    ])


EXTREMITY_ROWS = [4, 8, 12]


def encode_skeleton_as_offsets(s):
    return np.stack([
        s.spine2 - s.spine1,
        s.spine3 - s.spine2,
        s.neck - s.spine3,
        s.head - s.neck,
        s.head_top - s.head,
        s.l_hip - s.spine1,
        s.l_knee - s.l_hip,
        s.l_ankle - s.l_knee,
        s.l_foot - s.l_ankle,
        s.r_hip - s.spine1,
        s.r_knee - s.r_hip,
        s.r_ankle - s.r_knee,
        s.r_foot - s.r_ankle,
        s.l_collar - s.neck,
        s.l_shoulder - s.l_collar,
        s.l_elbow - s.l_shoulder,
        s.l_wrist - s.l_elbow,
        s.r_collar - s.neck,
        s.r_shoulder - s.r_collar,
        s.r_elbow - s.r_shoulder,
        s.r_wrist - s.r_elbow
    ])


def decode_skeleton_from_offsets(offsets, as_ndarray=False):
    def get(i):
        return offsets[i, :]

    spine2 = get(0)
    spine3 = spine2 + get(1)
    neck = spine3 + get(2)
    head = neck + get(3)
    head_top = head + get(4)

    l_hip = get(5)
    l_knee = l_hip + get(6)
    l_ankle = l_knee + get(7)
    l_foot = l_ankle + get(8)
    r_hip = get(9)
    r_knee = r_hip + get(10)
    r_ankle = r_knee + get(11)
    r_foot = r_ankle + get(12)

    l_collar = neck + get(13)
    l_shoulder = l_collar + get(14)
    l_elbow = l_shoulder + get(15)
    l_wrist = l_elbow + get(16)
    r_collar = neck + get(17)
    r_shoulder = r_collar + get(18)
    r_elbow = r_shoulder + get(19)
    r_wrist = r_elbow + get(20)

    skeleton = Skeleton(
        spine1=np.zeros(3), spine2=spine2, spine3=spine3,
        neck=neck, head=head, head_top=head_top,
        l_hip=l_hip, l_knee=l_knee, l_ankle=l_ankle, l_foot=l_foot,
        r_hip=r_hip, r_knee=r_knee, r_ankle=r_ankle, r_foot=r_foot,
        l_collar=l_collar, l_shoulder=l_shoulder, l_elbow=l_elbow,
        l_wrist=l_wrist,
        r_collar=r_collar, r_shoulder=r_shoulder, r_elbow=r_elbow,
        r_wrist=r_wrist
    )
    return np.stack(skeleton[1:]) if as_ndarray else skeleton


def load_amass_skeleton(pose, visualize):
    xyz = pose[:22, :].astype(np.float32)
    assert xyz.shape == (22, 3)

    spine1_raw = xyz[0, :]
    xyz -= spine1_raw

    if visualize:
        img = render_points(xyz[:, 0], xyz[:, 2])
        cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Neck is in a weird place
    forward_vec = get_canonical_orientation(
        xyz[[0, 3, 6, 13, 14, 16, 17], :],
        np.cross(xyz[13, :] - xyz[0, :], xyz[14, :] - xyz[0, :]), # left x Right
        (xyz[13, :] + xyz[14, :]) / 2 - xyz[0, :]  # neck - hip
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
        spine1=get(0), spine2=get(3), spine3=get(6),
        neck=(get(13) + get(14)) / 2, head=get(12), head_top=get(15),
        l_hip=get(2), l_knee=get(5), l_ankle=get(8), l_foot=get(11),
        r_hip=get(1), r_knee=get(4), r_ankle=get(7), r_foot=get(10),
        l_collar=get(14), l_shoulder=get(17), l_elbow=get(19), l_wrist=get(21),
        r_collar=get(13), r_shoulder=get(16), r_elbow=get(18), r_wrist=get(20)
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
    return (spine1_raw, theta, skeleton_offsets)

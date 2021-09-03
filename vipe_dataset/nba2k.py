from collections import namedtuple
import numpy as np
import cv2

from .util import render_points, get_canonical_orientation, flip_skeleton_offsets


_BaseSkeleton = namedtuple('_BaseSkeleton', [
    'hips',
    'rhip',
    'rknee',
    'rankle',
    'lhip',
    'lknee',
    'lankle',
    'spine',
    'neck',
    'head',
    'lshoulder',
    'lelbow',
    'lwrist',
    'rshoulder',
    'relbow',
    'rwrist',
    # 'rthumb',
    # 'rfore',
    # 'rmiddle',
    # 'rring',
    # 'rlittle',
    'rtoe',
    'rheel',
    'reye',
    'rear',
    # 'lthumb',
    # 'lfore',
    # 'lmiddle',
    # 'lring',
    # 'llittle',
    'ltoe',
    'lheel',
    'leye',
    'lear',
    'nose'
])

COLORS = tuple('brrrgggbbbgggrrrrrrrggggb')


class Skeleton(_BaseSkeleton):

    bones = (
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9),
        (8, 10), (10, 11), (11, 12),
        (8, 13), (13, 14), (14, 15),
        (3, 16), (3, 17),
        (9, 18), (18, 19),
        (6, 20), (6, 21),
        (9, 22), (9, 23),
        (9, 24)
    )


Z_UNIT = np.array([0., 0., 1.])

SCHEMA = """
# 0 pelvis
# 1 rhip
# 2 rknee
# 3 rankle
# 4 lhip
# 5 lknee
# 6 lankle
# 7 spine1
# 8 neck
# 9 head
# 10 lshoulder
# 11 lelbow
# 12 lwrist
# 13 rshoulder
# 14 relbow
# 15 rwrist
# 16 rthumb
# 17 rfore
# 18 rmiddle
# 19 rring
# 20 rlittle
# 21 rtoe
# 22 rheel
# 23 reye
# 24 rear
# 25 lthumb
# 26 lfore
# 27 lmiddle
# 28 lring
# 29 llittle
# 30 ltoe
# 31 lheel
# 32 leye
# 33 lear
# 34 nose
"""

XFLIP_ROWS = [3, 4, 5, 0, 1, 2, 6, 7, 8, 12, 13, 14, 9, 10, 11, 19, 20, 21, 22, 15, 16, 17, 18, 23]


def get_skeleton_parent_cossim(s):
    # Assume normed offsets
    def dot(i, j):
        return s[i, :].dot(s[j, :])
    return np.array([
        dot(6, 0), dot(0, 1), dot(1, 2),
        dot(6, 3), dot(3, 4), dot(4, 5),
        1, dot(6, 7), dot(7, 8),
        dot(7, 9), dot(9, 10), dot(10, 11),
        dot(7, 12), dot(12, 13), dot(13, 14),
        dot(2, 15), dot(2, 16),
        dot(8, 17), dot(17, 18),
        dot(5, 19), dot(5, 20),
        dot(8, 21), dot(21, 22),
        dot(8, 23)
    ])


EXTREMITY_ROWS = list(range(15, 24))


def encode_skeleton_as_offsets(s):
    return np.stack([
        s.rhip - s.hips,
        s.rknee - s.rhip,
        s.rankle - s.rknee,
        s.lhip - s.hips,
        s.lknee - s.lhip,
        s.lankle - s.lknee,
        s.spine - s.hips,
        s.neck - s.spine,
        s.head - s.neck,
        s.lshoulder - s.neck,
        s.lelbow - s.lshoulder,
        s.lwrist - s.lelbow,
        s.rshoulder - s.neck,
        s.relbow - s.rshoulder,
        s.rwrist - s.relbow,
        s.rtoe - s.rankle,
        s.rheel - s.rankle,
        s.reye - s.head,
        s.rear - s.reye,
        s.ltoe - s.lankle,
        s.lheel - s.lankle,
        s.leye - s.head,
        s.lear - s.leye,
        s.nose - s.head
    ])


def decode_skeleton_from_offsets(offsets, as_ndarray=False):
    def get(i):
        return offsets[i, :]

    rhip = get(0)
    rknee = rhip + get(1)
    rankle = rknee + get(2)
    lhip = get(3)
    lknee = lhip + get(4)
    lankle = lknee + get(5)
    spine = get(6)
    neck = spine + get(7)
    head = neck + get(8)
    lshoulder = neck + get(9)
    lelbow = lshoulder + get(10)
    lwrist = lelbow + get(11)
    rshoulder = neck + get(12)
    relbow = rshoulder + get(13)
    rwrist = relbow + get(14)
    rtoe = rankle + get(15)
    rheel = rankle + get(16)
    reye = head + get(17)
    rear = reye + get(18)
    ltoe = lankle + get(19)
    lheel = lankle + get(20)
    leye = head + get(21)
    lear = leye + get(22)
    nose = head + get(23)

    skeleton = Skeleton(
        hips=np.zeros(3), rhip=rhip, rknee=rknee, rankle=rankle, lhip=lhip,
        lknee=lknee, lankle=lankle, spine=spine, neck=neck, head=head,
        lshoulder=lshoulder, lelbow=lelbow, lwrist=lwrist, rshoulder=rshoulder,
        relbow=relbow, rwrist=rwrist,
        rtoe=rtoe, rheel=rheel, reye=reye, rear=rear,
        ltoe=ltoe, lheel=lheel, leye=leye, lear=lear,
        nose=nose
    )
    return np.stack(skeleton[1:]) if as_ndarray else skeleton


def load_nba2k_skeleton(pose, visualize):
    xyz = pose[:, [2, 0, 1]]
    assert xyz.shape == (35, 3)

    hips_raw = xyz[0, :]
    xyz -= hips_raw

    if visualize:
        img = render_points(xyz[:, 0], xyz[:, 2])
        cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    forward_vec = get_canonical_orientation(
        xyz[[0, 1, 4, 7, 8, 10, 13], :],
        np.cross(xyz[10, :] - xyz[0, :], xyz[13, :] - xyz[0, :]), # left x Right
        xyz[8, :] - xyz[0, :]  # neck - hip
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
        hips=get(0), rhip=get(1), rknee=get(2), rankle=get(3), lhip=get(4),
        lknee=get(5), lankle=get(6), spine=get(7), neck=get(8), head=get(9),
        lshoulder=get(10), lelbow=get(11), lwrist=get(12), rshoulder=get(13),
        relbow=get(14), rwrist=get(15),
        # rthumb=get(16), rfore=get(17), rmiddle=get(18), rring=get(19), rlittle=get(20),
        rtoe=get(21), rheel=get(22), reye=get(23), rear=get(24),
        # lthumb=get(25), lfore=get(26), lmiddle=get(27), lring=get(28), llittle=get(29),
        ltoe=get(30), lheel=get(31), leye=get(32), lear=get(33), nose=get(34)
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

import math
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.distance import pdist


MAX_NEG_SAMPLE_TRIES = 10
NEG_SAMPLE_JOINT_COS_THRESHOLD = math.cos(math.radians(45))
NEG_SAMPLE_STAT_FREQ = 1000


def _calc_mean_3d_kp_offset_norms(points, poses_3d):
    kp_offset_sum = None
    n = 0
    for k, _ in points:
        if not isinstance(k, tuple):
            k = tuple(k)
        for _, _, kp_offsets in poses_3d[k]:
            kp_offset_lengths = np.linalg.norm(kp_offsets, axis=1)
            if kp_offset_sum is None:
                kp_offset_sum = kp_offset_lengths
            else:
                kp_offset_sum += kp_offset_lengths
            n += 1
    return kp_offset_sum / n


def is_good_3d_neg_sample(a, b, ignore=None):
    dot = np.sum(a * b, axis=1)
    if ignore is not None:
        dot[ignore] = 1     # set these similarities to 1
    return np.min(dot) <= NEG_SAMPLE_JOINT_COS_THRESHOLD


def normalize_3d_offsets(kp_offsets):
    kp_dists = np.linalg.norm(kp_offsets, axis=1)
    kp_offsets = kp_offsets / kp_dists[:, None]
    return kp_offsets, kp_dists


def get_3d_features(
        abs_kp_offsets, skl_module,
        include_extremities=False, include_root_directions=True
):
    norm_kp_offsets = normalize_3d_offsets(abs_kp_offsets)[0]
    kp_features = [
        norm_kp_offsets,
        np.arccos(skl_module.get_skeleton_parent_cossim(norm_kp_offsets)
                  .reshape(-1, 1)) / np.pi - 0.5,
    ]
    if include_root_directions:
        kp_features.append(normalize_3d_offsets(
            skl_module.decode_skeleton_from_offsets(
                abs_kp_offsets, as_ndarray=True))[0])
    kp_features = np.hstack(kp_features)
    if not include_extremities:
        kp_features[skl_module.EXTREMITY_ROWS, :] = 0
    return kp_features


"""
"keypoints": {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
},
"""
NUM_COCO_KEYPOINTS_ORIG = 17

# Ignore eyes and ears
NUM_COCO_KEYPOINTS = 13
COCO_POINTS_IDXS = [0] + list(range(5, 17))

COCO_FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                    [13, 14], [15, 16]]
COCO_FLIP_IDXS = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
COCO_TORSO_POINTS = [5, 6, 11, 12]

COCO_BONES_ORIG = [(a - 1, b - 1) for a, b in [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5),
    (4, 6), (5, 7)]]
COCO_BONES = [
    x for x in COCO_BONES_ORIG
    if x[0] in COCO_POINTS_IDXS and x[1] in COCO_POINTS_IDXS]
NUM_COCO_BONES = len(COCO_BONES)


def normalize_2d_skeleton(kp, flip, to_tensor=True, zero_confs=False,
                          include_bone_features=False):
    kp = kp.copy()

    # Make the hips 0
    kp[:, :2] -= (kp[11, :2] + kp[12, :2]) / 2

    # Normalize distance from left-right sides
    max_torso_dist = np.max(pdist(kp[COCO_TORSO_POINTS, :2]))
    if max_torso_dist == 0:
        max_torso_dist = 1  # prevent 0div
    kp[:, :2] *= 0.5 / max_torso_dist

    if flip:
        kp = kp[COCO_FLIP_IDXS, :]
        kp[:, 0] *= -1

    if zero_confs:
        kp[:, 2] = 0
    else:
        # Shift confidences to -0.5, 0.5
        kp[:, 2] -= 0.5

    if include_bone_features:
        bones = np.zeros((len(COCO_BONES), 3))
        for i, (a, b) in enumerate(COCO_BONES):
            bones[i, :2] = kp[a, :2] - kp[b, :2]
            bones[i, 2] = (kp[a, 2] + kp[b, 2]) / 2

    kp = kp[COCO_POINTS_IDXS, :]
    if include_bone_features:
        kp = np.vstack((kp, bones))
    return torch.FloatTensor(kp) if to_tensor else kp


class D3KeypointDataset(Dataset):

    def __init__(self, points, poses_3d, random_hflip, augment_camera,
                 embed_bones, target_len, debug_info=False):
        super().__init__()

        self.points = points
        self.poses_3d = poses_3d
        self.embed_bones = embed_bones

        self.augment_camera = augment_camera
        self.camera_aug_prob = 0.5

        self.random_hflip = random_hflip
        self.scale = math.ceil(target_len / len(points))
        self.debug_info = debug_info

        # Stats for negative sampling
        self.sample_count = 0
        self.neg_sample_fail_count = 0

    def _should_flip(self):
        return self.random_hflip and random.getrandbits(1) > 0

    def __len__(self):
        return len(self.points) * self.scale

    def get(self, index):
        return self.points[index % len(self.points)]

    def _log_neg_sample_fail(self):
        self.neg_sample_fail_count += 1
        if self.neg_sample_fail_count % NEG_SAMPLE_STAT_FREQ == 0:
            print('Neg sample fail rate: {}'.format(
                self.neg_sample_fail_count / self.sample_count))

    def _should_project(self):
        return self.augment_camera and random.random() < self.camera_aug_prob

    @property
    def mean_kp_offset_norms(self):
        return _calc_mean_3d_kp_offset_norms(self.points, self.poses_3d)

    @property
    def num_sequences(self):
        return len(self.points)
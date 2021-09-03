import os
import math
import random
from collections import Counter, defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from . import human36m, people3d, nba2k, amass
from .util import flip_skeleton_offsets
from .dataset_base import (
    D3KeypointDataset, MAX_NEG_SAMPLE_TRIES, is_good_3d_neg_sample,
    normalize_3d_offsets, normalize_2d_skeleton, get_3d_features,
    NUM_COCO_KEYPOINTS_ORIG)
from util.io import load_pickle, load_gz_json


USE_EXTREMITIES = True
USE_ROOT_DIRECTIONS = True

CAMERA_AUG_ELEVATION_RANGE = (-np.pi / 6, np.pi / 6)
CAMERA_AUG_ROLL_RANGE = (-np.pi / 6, np.pi / 6)


def _random_project_3d(coco_xyz, elevation=None, roll=None):
    # Point the camera at the torso
    # coco_xyz -= np.mean([
    #     skl.left_arm, skl.right_arm, skl.left_up_leg, skl.right_up_leg],
    #     axis=0)

    # Rotate around z
    a = np.random.uniform(-np.pi, np.pi)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    rot_z_t = np.array([
        [cos_a, sin_a, 0],
        [-sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    coco_xyz = coco_xyz.dot(rot_z_t)

    if elevation is not None:
        # Rotate around x
        b = np.random.uniform(*elevation)
        cos_b = math.cos(b)
        sin_b = math.sin(b)
        rot_x_t = np.array([
            [1, 0, 0],
            [0, cos_b, sin_b],
            [0, -sin_b, cos_b],
        ])
        coco_xyz = coco_xyz.dot(rot_x_t)

    if roll is not None:
        # Rotate around y
        c = np.random.uniform(*roll)
        cos_c = math.cos(c)
        sin_c = math.sin(c)
        rot_y_t = np.array([
            [cos_c, 0, sin_c],
            [0, 1, 0],
            [-sin_c, 0, cos_c],
        ])
        coco_xyz = coco_xyz.dot(rot_y_t)

    # Randomize confidence scores
    conf = np.random.uniform(0.5, 1, size=NUM_COCO_KEYPOINTS_ORIG)
    conf[1:5] = 0

    # Project into 2D
    coco_xzc = np.hstack((coco_xyz[:, [0, 2]], conf[:, None]))

    # Invert z to convert to pixel coordinates
    coco_xzc[:, 1] *= -1

    assert coco_xzc.shape == (NUM_COCO_KEYPOINTS_ORIG, 3)
    return coco_xzc


def _sample_camera_pair(all_cameras_and_2d_poses):
    if len(all_cameras_and_2d_poses) > 1:
        views = np.random.choice(
            range(len(all_cameras_and_2d_poses)), 2, replace=False)
    else:
        views = (0, 0)
    camera1, pose_2d1 = all_cameras_and_2d_poses[views[0]]
    camera2, pose_2d2 = all_cameras_and_2d_poses[views[1]]
    return camera1, camera2, pose_2d1, pose_2d2


class Human36MDataset(D3KeypointDataset):

    def get_sequence(self, index, camera=None, stride=25):
        (person, action), frames = self.get(index)
        seq_poses = self.poses_3d[(person, action)]

        sequence = []
        for i, (frame_num, all_cameras_and_2d_poses) in enumerate(frames):
            if i % stride != 0:
                continue

            if camera is None:
                # Choose a random camera
                camera, pose2d = random.choice(all_cameras_and_2d_poses)
            else:
                for camera2, pose2d in all_cameras_and_2d_poses:
                    if camera2 == camera:
                        break
                else:
                    continue

            # Load 3d ground truth
            if frame_num >= len(seq_poses):
                print('Invalid frame: {} > {} (max_frame: {})'.format(
                    frame_num, len(seq_poses), frames[-1][0]))
                break
            _, rotation, abs_kp_offsets = seq_poses[frame_num]
            norm_kp_offsets, kp_dists = normalize_3d_offsets(abs_kp_offsets)

            sequence.append({
                'person': person,
                'action': action,
                'frame': frame_num,
                'rotation': rotation,
                'kp_offsets': norm_kp_offsets,
                'kp_offset_norms': kp_dists,
                'camera': camera,
                'pose': normalize_2d_skeleton(
                    pose2d, False, include_bone_features=self.embed_bones)
            })
        return sequence

    @staticmethod
    def _random_project_3d(raw_kp_offsets):
        skl = human36m.decode_skeleton_from_offsets(raw_kp_offsets)
        coco_xyz = np.stack([
            skl.nose,
            skl.nose, # No eyes in h36m
            skl.nose,
            skl.nose, # No ears in h36m
            skl.nose,
            skl.left_arm,
            skl.right_arm,
            skl.left_forearm,
            skl.right_forearm,
            skl.left_hand,
            skl.right_hand,
            skl.left_up_leg,
            skl.right_up_leg,
            skl.left_leg,
            skl.right_leg,
            skl.left_foot,
            skl.right_foot,
        ])
        return _random_project_3d(
            coco_xyz, elevation=CAMERA_AUG_ELEVATION_RANGE,
            roll=CAMERA_AUG_ROLL_RANGE)

    def _get_negative_sample(self, frames, seq_poses, norm_kp_offsets):
        # Try to find a pose in the same sequence that differs the current one
        neg_flip = False
        for _ in range(MAX_NEG_SAMPLE_TRIES):
            neg_frame_num, neg_cameras_and_2d_poses = random.choice(frames)
            if neg_frame_num >= len(seq_poses):
                continue
            neg_raw_offsets = seq_poses[neg_frame_num][-1]
            neg_flip = self._should_flip()
            if is_good_3d_neg_sample(
                    normalize_3d_offsets(
                        flip_skeleton_offsets(neg_raw_offsets, human36m.XFLIP_ROWS)
                        if neg_flip else neg_raw_offsets
                    )[0],
                    norm_kp_offsets,
                    ignore=None if USE_EXTREMITIES else human36m.EXTREMITY_ROWS
            ):
                if self._should_project():
                    # Need to project with 3d before flipping
                    neg_pose2d = Human36MDataset._random_project_3d(neg_raw_offsets)
                else:
                    neg_pose2d = random.choice(neg_cameras_and_2d_poses)[1]
                break
        else:
            neg_pose2d = None
            self._log_neg_sample_fail()
        return neg_pose2d, neg_flip

    def __getitem__(self, index):
        self.sample_count += 1

        (person, action), frames = self.get(index)
        seq_poses = self.poses_3d[(person, action)]
        flip = self._should_flip()

        while True:
            frame_num, all_cameras_and_2d_poses = random.choice(frames)
            if frame_num < len(seq_poses):
                break
        assert len(all_cameras_and_2d_poses) > 0

        # Load 3d ground truth
        _, rotation, raw_kp_offsets = seq_poses[frame_num]

        # Flip and normalize 3D
        abs_kp_offsets = raw_kp_offsets
        if flip:
            rotation = -rotation
            abs_kp_offsets = flip_skeleton_offsets(
                abs_kp_offsets, human36m.XFLIP_ROWS)

        # Sample two random cameras
        camera1, camera2, pose_2d1, pose_2d2 = _sample_camera_pair(
            all_cameras_and_2d_poses)

        # Replace with random projections if enabled
        if self._should_project():
            camera1 = ''
            pose_2d1 = Human36MDataset._random_project_3d(raw_kp_offsets)
        if self._should_project():
            camera2 = ''
            pose_2d2 = Human36MDataset._random_project_3d(raw_kp_offsets)

        # Negative sample
        neg_pose2d, neg_flip = self._get_negative_sample(
            frames, seq_poses, normalize_3d_offsets(abs_kp_offsets)[0])

        norm_pose1 = normalize_2d_skeleton(
            pose_2d1, flip, include_bone_features=self.embed_bones)
        ret = {
            'kp_features': get_3d_features(
                abs_kp_offsets, human36m, include_extremities=USE_EXTREMITIES,
                include_root_directions=USE_ROOT_DIRECTIONS),
            'pose1': norm_pose1,
            'pose2': normalize_2d_skeleton(
                pose_2d2, flip, include_bone_features=self.embed_bones),
            'pose_neg': torch.zeros_like(norm_pose1) if neg_pose2d is None
                        else normalize_2d_skeleton(
                            neg_pose2d, neg_flip,
                            include_bone_features=self.embed_bones),
            'pose_neg_is_valid': int(neg_pose2d is not None)
        }
        if self.debug_info:
            ret.update({
                'person': person,
                'action': action,
                'frame': frame_num,
                'rotation': rotation,
                'camera1': camera1,
                'camera2': camera2,
                'is_flip': flip
            })
        return ret

    @staticmethod
    def load_default(pose_2d_dir, pose_3d_file, embed_bones, augment_camera):
        # These do not have 3d poses
        exclude_actions = {'_ALL', '_ALL 1'}

        pose_2d = defaultdict(lambda: defaultdict(list))
        for pose_2d_file in tqdm(
                os.listdir(pose_2d_dir), desc='Loading human3.6m'
        ):
            person, action, camera, _ = pose_2d_file.split('.', 3)
            if action in exclude_actions:
                continue
            seq_pose = load_gz_json(os.path.join(pose_2d_dir, pose_2d_file))
            for frame, pose_data in seq_pose:
                if len(pose_data) > 0:
                    kp = np.array(pose_data[0][-1], dtype=np.float32)
                    pose_2d[(person, action)][frame].append((camera, kp))
        pose_2d = [(k, list(v.items())) for k, v in pose_2d.items()]
        pose_3d = load_pickle(pose_3d_file)

        all_people = {x[0][0] for x in pose_2d}
        val_people = {'S9', 'S11'}
        print('{} / {} people reserved for validation'.format(
            len(val_people), len(all_people)))
        assert val_people <= all_people

        train_2d = [x for x in pose_2d if x[0][0] not in val_people]
        train_2d.sort()
        train_dataset = Human36MDataset(
            train_2d, pose_3d, True, augment_camera, embed_bones, 20000)

        val_2d = [x for x in pose_2d if x[0][0] in val_people]
        val_2d.sort()
        val_dataset = Human36MDataset(
            val_2d, pose_3d, True, augment_camera, embed_bones, 2000)
        return train_dataset, val_dataset

# Common format for amass, 3dpeople, and nba2k
def _load_person_poses(pose_2d_dir, pose_2d_file):
    person_pose = []
    for frame, all_camera_pose_data in sorted(
            load_gz_json(os.path.join(pose_2d_dir, pose_2d_file))
    ):
        frame_camera_pose = []
        for camera, pose_data in all_camera_pose_data:
            assert len(pose_data) > 0
            if len(pose_data) > 0:
                kp = np.array(pose_data[-1], dtype=np.float32)
                frame_camera_pose.append((camera, kp))
        person_pose.append((frame, frame_camera_pose))
    assert len(person_pose) > 0
    return person_pose


class NBA2kDataset(D3KeypointDataset):

    CAMERA_AUG_PROB = 0.5
    CAMERA_AUG_ELEVATION_RANGE = (-np.pi / 6, np.pi / 6)

    @staticmethod
    def _random_project_3d(raw_kp_offsets):
        skl = nba2k.decode_skeleton_from_offsets(raw_kp_offsets)
        coco_xyz = np.stack([
            skl.nose,
            skl.leye,
            skl.reye,
            skl.lear,
            skl.rear,
            skl.lshoulder,
            skl.rshoulder,
            skl.lelbow,
            skl.relbow,
            skl.lwrist,
            skl.rwrist,
            skl.lhip,
            skl.rhip,
            skl.lknee,
            skl.rknee,
            skl.lankle,
            skl.rankle,
        ])
        return _random_project_3d(
            coco_xyz, elevation=CAMERA_AUG_ELEVATION_RANGE,
            roll=CAMERA_AUG_ROLL_RANGE)

    def get_sequence(self, index, camera=None, stride=4):
        person_key, frame_data = self.get(index)
        person_3d_poses = self.poses_3d[person_key]

        sequence = []
        for i, (frame_num, all_cameras_and_poses) in enumerate(frame_data):
            if i % stride != 0:
                continue

            # Load 3d ground truth
            _, rotation, abs_kp_offsets = person_3d_poses[frame_num]
            norm_kp_offsets, kp_dists = normalize_3d_offsets(abs_kp_offsets)

            sequence.append({
                'person': person_key[0],
                'action': '',
                'camera': '',
                'frame': frame_num,
                'rotation': rotation,
                'kp_offsets': norm_kp_offsets,
                'kp_offset_norms': kp_dists,
                'pose': normalize_2d_skeleton(
                    all_cameras_and_poses[0][-1], False,
                    include_bone_features=self.embed_bones)
            })
        return sequence

    def _get_negative_sample(self, frame_data, seq_poses, norm_kp_offsets):
        # Try to find a pose in the same sequence that differs the current one
        neg_flip = False
        for _ in range(MAX_NEG_SAMPLE_TRIES):
            neg_frame_num, _ = random.choice(frame_data)
            neg_raw_offsets = seq_poses[neg_frame_num][-1]
            neg_flip = self._should_flip()
            if is_good_3d_neg_sample(
                    normalize_3d_offsets(
                        flip_skeleton_offsets(neg_raw_offsets, nba2k.XFLIP_ROWS)
                        if neg_flip else neg_raw_offsets
                    )[0],
                    norm_kp_offsets,
                    ignore=None if USE_EXTREMITIES else nba2k.EXTREMITY_ROWS
            ):
                neg_pose2d = NBA2kDataset._random_project_3d(neg_raw_offsets)
                break
        else:
            neg_pose2d = None
            self._log_neg_sample_fail()
        return neg_pose2d, neg_flip

    def __getitem__(self, index):
        self.sample_count += 1

        person_key, frame_data = self.get(index)
        person_3d_poses = self.poses_3d[person_key]
        frame_num, all_cameras_and_poses = random.choice(frame_data)
        pose_2d = all_cameras_and_poses[0][-1]
        flip = self._should_flip()

        # Load 3d ground truth
        _, rotation, raw_kp_offsets = person_3d_poses[frame_num]

        # Flip and normalize 3D
        abs_kp_offsets = raw_kp_offsets
        if flip:
            rotation = -rotation
            abs_kp_offsets = flip_skeleton_offsets(
                abs_kp_offsets, nba2k.XFLIP_ROWS)

        if self._should_project():
            pose_2d = NBA2kDataset._random_project_3d(raw_kp_offsets)

        ret = {'kp_features': get_3d_features(
                    abs_kp_offsets, nba2k, include_extremities=USE_EXTREMITIES,
                    include_root_directions=USE_ROOT_DIRECTIONS),
                'pose1': normalize_2d_skeleton(
                    pose_2d, flip, include_bone_features=self.embed_bones)}

        if self.augment_camera:
            pose_2d2 = NBA2kDataset._random_project_3d(raw_kp_offsets)
            ret['pose2'] = normalize_2d_skeleton(
                    pose_2d2, flip, include_bone_features=self.embed_bones)

            neg_pose2d, neg_flip = self._get_negative_sample(
                frame_data, person_3d_poses,
                normalize_3d_offsets(abs_kp_offsets)[0])
            ret['pose_neg'] = (
                torch.zeros_like(pose_2d) if neg_pose2d is None else
                normalize_2d_skeleton(neg_pose2d, neg_flip,
                                      include_bone_features=self.embed_bones))
            ret['pose_neg_is_valid'] = int(neg_pose2d is not None)

        if self.debug_info:
            ret.update({
                'person': person_key[0],
                'action': '',
                'frame': frame_num,
                'rotation': rotation,
                'camera1': '', 'camera2': '',
                'is_flip': flip
            })
        return ret

    @staticmethod
    def load_default(pose_2d_dir, pose_3d_file, embed_bones):
        pose_3d = load_pickle(pose_3d_file)
        pose_2d = []
        for pose_2d_file in tqdm(os.listdir(pose_2d_dir), desc='Loading NBA2K'):
            person = pose_2d_file.split('.', 1)[0]
            pose_2d.append((
                (person,), _load_person_poses(pose_2d_dir, pose_2d_file)))

        all_people = {x[0][0] for x in pose_2d}
        val_people = {'alfred', 'allen', 'barney', 'bradley'}
        print('{} / {} people reserved for validation'.format(
            len(val_people), len(all_people)))
        assert val_people <= all_people

        train_2d = [x for x in pose_2d if x[0][0] not in val_people]
        train_2d.sort()
        train_dataset = NBA2kDataset(
            train_2d, pose_3d, True, True, embed_bones, 5000)

        val_2d = [x for x in pose_2d if x[0][0] in val_people]
        val_2d.sort()
        val_dataset = NBA2kDataset(
            val_2d, pose_3d, True, True, embed_bones, 500)
        return train_dataset, val_dataset


class People3dDataset(D3KeypointDataset):

    def get_sequence(self, index, camera=None, stride=2):
        (person, action), frame_data = self.get(index)
        seq_poses = self.poses_3d[(person, action)]

        sequence = []
        for i, (frame_num, all_cameras_and_2d_poses) in enumerate(frame_data):
            if i % stride != 0:
                continue

            if camera is None:
                # Choose a random camera
                camera, pose2d = random.choice(all_cameras_and_2d_poses)
            else:
                for camera2, pose2d in all_cameras_and_2d_poses:
                    if camera2 == camera:
                        break
                else:
                    continue

            # Load 3d ground truth
            _, rotation, abs_kp_offsets = seq_poses[frame_num - 1]
            norm_kp_offsets, kp_dists = normalize_3d_offsets(abs_kp_offsets)

            sequence.append({
                'person': person,
                'action': action,
                'frame': frame_num,
                'rotation': rotation,
                'kp_offsets': norm_kp_offsets,
                'kp_offset_norms': kp_dists,
                'camera': camera,
                'pose': normalize_2d_skeleton(
                    pose2d, False, include_bone_features=self.embed_bones)
            })
        return sequence

    @staticmethod
    def _random_project_3d(raw_kp_offsets):
        skl = people3d.decode_skeleton_from_offsets(raw_kp_offsets)
        coco_xyz = np.stack([
            (skl.head + skl.left_eye + skl.right_eye) / 3,
            skl.left_eye,
            skl.right_eye,
            skl.left_eye, # No ears in 3dpeople
            skl.right_eye,
            skl.left_arm,
            skl.right_arm,
            skl.left_forearm,
            skl.right_forearm,
            skl.left_hand,
            skl.right_hand,
            skl.left_up_leg,
            skl.right_up_leg,
            skl.left_leg,
            skl.right_leg,
            skl.left_foot,
            skl.right_foot,
        ])
        return _random_project_3d(
            coco_xyz, elevation=CAMERA_AUG_ELEVATION_RANGE,
            roll=CAMERA_AUG_ROLL_RANGE)

    def _get_negative_sample(self, frame_data, seq_poses, norm_kp_offsets):
        # Try to find a pose in the same sequence that differs the current one
        neg_flip = False
        for _ in range(MAX_NEG_SAMPLE_TRIES):
            neg_frame_num, neg_cameras_and_2d_poses = random.choice(frame_data)
            neg_raw_offsets = seq_poses[neg_frame_num - 1][-1]
            neg_flip = self._should_flip()
            if is_good_3d_neg_sample(
                    normalize_3d_offsets(
                        flip_skeleton_offsets(neg_raw_offsets, people3d.XFLIP_ROWS)
                        if neg_flip else neg_raw_offsets
                    )[0],
                    norm_kp_offsets,
                    ignore=None if USE_EXTREMITIES else people3d.EXTREMITY_ROWS
            ):
                if self._should_project():
                    neg_pose2d = People3dDataset._random_project_3d(neg_raw_offsets)
                else:
                    neg_pose2d = random.choice(neg_cameras_and_2d_poses)[1]
                break
        else:
            neg_pose2d = None
            self._log_neg_sample_fail()
        return neg_pose2d, neg_flip

    def __getitem__(self, index):
        self.sample_count += 1

        (person, action), frame_data = self.get(index)
        seq_poses = self.poses_3d[(person, action)]
        flip = self._should_flip()

        frame_num, all_cameras_and_2d_poses = random.choice(frame_data)
        assert len(all_cameras_and_2d_poses) > 0

        # Load 3d ground truth
        _, rotation, raw_kp_offsets = seq_poses[frame_num - 1]

        # Flip and normalize 3D
        abs_kp_offsets = raw_kp_offsets
        if flip:
            rotation = -rotation
            abs_kp_offsets = flip_skeleton_offsets(
                abs_kp_offsets, people3d.XFLIP_ROWS)

        # Sample two random cameras
        camera1, camera2, pose_2d1, pose_2d2 = _sample_camera_pair(
            all_cameras_and_2d_poses)

        # Replace with projections if needed
        if self._should_project():
            camera1 = ''
            pose_2d1 = People3dDataset._random_project_3d(raw_kp_offsets)
        if self._should_project():
            camera2 = ''
            pose_2d2 = People3dDataset._random_project_3d(raw_kp_offsets)

        # Get negative sample
        neg_pose2d, neg_flip = self._get_negative_sample(
            frame_data, seq_poses, normalize_3d_offsets(abs_kp_offsets)[0])

        norm_pose1 = normalize_2d_skeleton(
            pose_2d1, flip, include_bone_features=self.embed_bones)
        ret = {
            'kp_features': get_3d_features(
                abs_kp_offsets, people3d, include_extremities=USE_EXTREMITIES,
                include_root_directions=USE_ROOT_DIRECTIONS),
            'pose1': norm_pose1,
            'pose2': normalize_2d_skeleton(
                pose_2d2, flip, include_bone_features=self.embed_bones),
            'pose_neg': torch.zeros_like(norm_pose1) if neg_pose2d is None
                        else normalize_2d_skeleton(
                            neg_pose2d, neg_flip,
                            include_bone_features=self.embed_bones),
            'pose_neg_is_valid': int(neg_pose2d is not None)
        }
        if self.debug_info:
            ret.update({
                'person': person,
                'action': action,
                'frame': frame_num,
                'rotation': rotation,
                'camera1': camera1,
                'camera2': camera2,
                'is_flip': flip
            })
        return ret

    @staticmethod
    def load_default(pose_2d_dir, pose_3d_file, embed_bones, augment_camera):
        pose_2d = []
        for pose_2d_file in tqdm(
                os.listdir(pose_2d_dir), desc='Loading 3D people'
        ):
            person, action = pose_2d_file.split('.', 1)[0].split('__', 1)
            pose_2d.append(((person, action),
                             _load_person_poses(pose_2d_dir, pose_2d_file)))
        pose_3d = load_pickle(pose_3d_file)

        all_people = {x[0][0] for x in pose_2d}
        val_people = set()
        for s in ['man', 'woman']:
            val_people.update(['{}{:02d}'.format(s, i + 1) for i in range(4)])
        print('{} / {} people reserved for validation'.format(
            len(val_people), len(all_people)))
        assert val_people <= all_people

        train_2d = [x for x in pose_2d if x[0][0] not in val_people]
        train_2d.sort()
        train_dataset = People3dDataset(
            train_2d, pose_3d, True, augment_camera, embed_bones, 5000)

        val_2d = [x for x in pose_2d if x[0][0] in val_people]
        val_2d.sort()
        val_dataset = People3dDataset(
            val_2d, pose_3d, True, augment_camera, embed_bones, 500)
        return train_dataset, val_dataset


class AmassDataset(D3KeypointDataset):

    CAMERA_AUG_ELEVATION_RANGE = (-np.pi / 6, np.pi / 6)

    idx_stride = 25

    sample_weights = {
        'ACCAD': 1,
        'BMLhandball': 1,
        'BMLmovi': 1,
        'BMLrub': 1,
        'CMU': 1,
        'DFaust67': 1,
        'EKUT': 1,
        'EyesJapanDataset': 1,
        'HumanEva': 1,
        'KIT': 1,
        'MPIHDM05': 10,
        'MPILimits': 10,
        'MPImosh': 10,
        'SFU': 1,
        'SSMsynced': 1,
        'TCDhandMocap': 1,
        'TotalCapture': 1,
        'Transitionsmocap': 1
    }

    @staticmethod
    def _idx(frame_num):
        return frame_num // AmassDataset.idx_stride

    def get_sequence(self, index, camera=None, stride=25):
        (dataset, action), frame_data = self.get(index)
        seq_poses = self.poses_3d[(dataset, action)]

        sequence = []
        for i, (frame_num, all_cameras_and_2d_poses) in enumerate(frame_data):
            if i % stride != 0:
                continue

            camera, pose2d = random.choice(all_cameras_and_2d_poses)

            # Load 3d ground truth
            _, rotation, abs_kp_offsets = seq_poses[AmassDataset._idx(frame_num)]
            norm_kp_offsets, kp_dists = normalize_3d_offsets(abs_kp_offsets)

            sequence.append({
                'person': dataset,
                'action': action,
                'frame': frame_num,
                'rotation': rotation,
                'kp_offsets': norm_kp_offsets,
                'kp_offset_norms': kp_dists,
                'camera': camera,
                'pose': normalize_2d_skeleton(
                    pose2d, False, include_bone_features=self.embed_bones)
            })
        return sequence

    @staticmethod
    def _random_project_3d(raw_kp_offsets):
        skl = amass.decode_skeleton_from_offsets(raw_kp_offsets)
        nose = (skl.head_top + skl.head) / 2
        coco_xyz = np.stack([
            nose,
            nose, # No eyes in amass
            nose,
            nose, # No ears in amass
            nose,
            skl.l_shoulder,
            skl.r_shoulder,
            skl.l_elbow,
            skl.r_elbow,
            skl.l_wrist,
            skl.r_wrist,
            skl.l_hip,
            skl.r_hip,
            skl.l_knee,
            skl.r_knee,
            skl.l_ankle,
            skl.r_ankle,
        ])
        return _random_project_3d(
            coco_xyz, elevation=CAMERA_AUG_ELEVATION_RANGE,
            roll=CAMERA_AUG_ROLL_RANGE)

    def _get_negative_sample(self, frame_data, seq_poses, norm_kp_offsets):
        # Try to find a pose in the same sequence that differs the current one
        neg_flip = False
        for _ in range(MAX_NEG_SAMPLE_TRIES):
            neg_frame_num, neg_cameras_and_2d_poses = random.choice(frame_data)
            neg_raw_offsets = seq_poses[AmassDataset._idx(neg_frame_num)][-1]
            neg_flip = self._should_flip()
            if is_good_3d_neg_sample(
                    normalize_3d_offsets(
                        flip_skeleton_offsets(neg_raw_offsets, amass.XFLIP_ROWS)
                        if neg_flip else neg_raw_offsets
                    )[0],
                    norm_kp_offsets,
                    ignore=None if USE_EXTREMITIES else amass.EXTREMITY_ROWS
            ):
                if self._should_project():
                    neg_pose2d = AmassDataset._random_project_3d(neg_raw_offsets)
                else:
                    neg_pose2d = random.choice(neg_cameras_and_2d_poses)[1]
                break
        else:
            neg_pose2d = None
            self._log_neg_sample_fail()
        return neg_pose2d, neg_flip

    def __getitem__(self, index):
        self.sample_count += 1

        (dataset, action), frame_data = self.get(index)
        seq_poses = self.poses_3d[(dataset, action)]
        flip = self._should_flip()

        frame_num, all_cameras_and_2d_poses = random.choice(frame_data)
        assert len(all_cameras_and_2d_poses) > 0

        # Load 3d ground truth
        _, rotation, raw_kp_offsets = seq_poses[AmassDataset._idx(frame_num)]

        # Flip and normalize 3D
        abs_kp_offsets = raw_kp_offsets
        if flip:
            rotation = -rotation
            abs_kp_offsets = flip_skeleton_offsets(
                abs_kp_offsets, amass.XFLIP_ROWS)

        # Sample two random cameras
        camera1, camera2, pose_2d1, pose_2d2 = _sample_camera_pair(
            all_cameras_and_2d_poses)

        # Replace with projections if needed
        if self._should_project():
            camera1 = ''
            pose_2d1 = AmassDataset._random_project_3d(raw_kp_offsets)
        if self._should_project():
            camera2 = ''
            pose_2d2 = AmassDataset._random_project_3d(raw_kp_offsets)

        # Get negative sample
        neg_pose2d, neg_flip = self._get_negative_sample(
            frame_data, seq_poses, normalize_3d_offsets(abs_kp_offsets)[0])

        norm_pose1 = normalize_2d_skeleton(
            pose_2d1, flip, include_bone_features=self.embed_bones)
        ret = {
            'kp_features': get_3d_features(
                abs_kp_offsets, amass, include_extremities=USE_EXTREMITIES,
                include_root_directions=USE_ROOT_DIRECTIONS),
            'pose1': norm_pose1,
            'pose2': normalize_2d_skeleton(
                pose_2d2, flip, include_bone_features=self.embed_bones),
            'pose_neg': torch.zeros_like(norm_pose1) if neg_pose2d is None
                        else normalize_2d_skeleton(
                            neg_pose2d, neg_flip,
                            include_bone_features=self.embed_bones),
            'pose_neg_is_valid': int(neg_pose2d is not None)
        }
        if self.debug_info:
            ret.update({
                'person': dataset,
                'action': action,
                'frame': frame_num,
                'rotation': rotation,
                'camera1': camera1,
                'camera2': camera2,
                'is_flip': flip
            })
        return ret

    @staticmethod
    def load_default(pose_2d_dir, pose_3d_file, embed_bones, augment_camera):
        pose_2d = []
        for pose_2d_file in tqdm(
                os.listdir(pose_2d_dir), desc='Loading AMASS'
        ):
            dataset, action = pose_2d_file.split('.', 1)[0].split('_', 1)
            pose_2d.append(((dataset, action),
                             _load_person_poses(pose_2d_dir, pose_2d_file)))
        pose_3d = load_pickle(pose_3d_file)

        # Stride over subsampled datasets
        dataset_counter = Counter()

        all_datasets = set()
        all_sequences = []
        for item in pose_2d:
            dataset = item[0][0]
            dataset_weight = AmassDataset.sample_weights[dataset]
            if dataset_weight >= 1:
                for _ in range(round(dataset_weight)):
                    all_sequences.append(item)
            else:
                if dataset_counter[dataset] % round(1 / dataset_weight) == 0:
                    all_sequences.append(item)
                dataset_counter[dataset] += 1
            all_datasets.add(dataset)

        val_datasets = {'EyesJapanDataset'}
        print('{} / {} datasets reserved for validation'.format(
            len(val_datasets), len(all_datasets)))
        assert val_datasets <= all_datasets

        train_2d = [x for x in pose_2d if x[0][0] not in val_datasets]
        train_2d.sort()
        train_dataset = AmassDataset(
            train_2d, pose_3d, True, augment_camera, embed_bones, 20000)

        val_2d = [x for x in pose_2d if x[0][0] in val_datasets]
        val_2d.sort()
        val_dataset = AmassDataset(
            val_2d, pose_3d, True, augment_camera, embed_bones, 2000)
        return train_dataset, val_dataset


class Pairwise_People3dDataset(Dataset):

    def __init__(self, pose_2d, scale, embed_bones, random_hflip=True):
        super().__init__()
        self.random_hflip = random_hflip
        self.embed_bones = embed_bones

        # (person, action) -> frames
        self.point_dict = {
            tuple(a): (
                [x[0] for x in b],      # Frame list
                dict(b)                 # Frame to cameras
            ) for a, b in pose_2d
        }
        self.people = list(set(x[0] for x in self.point_dict))
        self.actions = list(set(x[1] for x in self.point_dict))
        self.scale = scale

    def __len__(self):
        return len(self.actions) * self.scale

    def _should_flip(self):
        return self.random_hflip and random.getrandbits(1) > 0

    def __getitem__(self, index):
        action = self.actions[index % len(self.actions)]
        person1, person2 = np.random.choice(
            self.people, 2, replace=False).tolist()

        frames1, frame_to_cameras1 = self.point_dict[(person1, action)]
        _, frame_to_cameras2 = self.point_dict[(person2, action)]
        for i in range(1000):
            if i == 10:
                print('Why is this taking so many tries?',
                      person1, person2, action)
            frame_num = random.choice(frames1)

            # No camera is available for person2; try again
            all_cameras2 = frame_to_cameras2.get(frame_num)
            if all_cameras2 is None:
                continue

            # Sample the cameras
            pose_2d1 = random.choice(frame_to_cameras1[frame_num])[1]
            pose_2d2 = random.choice(all_cameras2)[1]
            break
        else:
            raise RuntimeError('This dataset is really borked...')

        flip = self._should_flip()
        return {
            'pose1': normalize_2d_skeleton(
                pose_2d1, flip, include_bone_features=self.embed_bones),
            'pose2': normalize_2d_skeleton(
                pose_2d2, flip, include_bone_features=self.embed_bones),
            'is_same': True, 'is_flip': flip
        }

    @staticmethod
    def load_default(pose_2d_dir, scale, embed_bones):
        pose_2d = []
        for pose_2d_file in tqdm(
                os.listdir(pose_2d_dir), desc='Loading 3D people (Pairs)'
        ):
            person, action = pose_2d_file.split('.', 1)[0].split('__', 1)
            pose_2d.append(((person, action),
                             _load_person_poses(pose_2d_dir, pose_2d_file)))

        all_people = {x[0][0] for x in pose_2d}
        val_people = set()
        for s in ['man', 'woman']:
            val_people.update(['{}{:02d}'.format(s, i + 1) for i in range(4)])
        print('{} / {} people reserved for validation'.format(
            len(val_people), len(all_people)))
        assert val_people <= all_people

        train_2d = [x for x in pose_2d if x[0][0] not in val_people]
        train_2d.sort()
        train_dataset = Pairwise_People3dDataset(train_2d, scale, embed_bones)

        val_2d = [x for x in pose_2d if x[0][0] in val_people]
        val_2d.sort()
        val_dataset = Pairwise_People3dDataset(
            val_2d, int(scale * 0.2), embed_bones)
        return train_dataset, val_dataset
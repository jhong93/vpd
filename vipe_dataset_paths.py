from os.path import join

# Paths for 3D data
DATA_DIR = 'data/vipe'

PEOPLE_3D_MANIFEST_FILE = join(DATA_DIR, '3dpeople', 'pose_manifest.json')
PEOPLE_3D_3D_POSE_FILE = join(DATA_DIR, '3dpeople', 'ground_truth_3d_pose.pkl')
PEOPLE_3D_KEYPOINT_DIR = join(DATA_DIR, '3dpeople', 'cocopose')

HUMAN36M_MANIFEST_FILE = join(DATA_DIR, 'human3.6m', 'pose_manifest.json')
HUMAN36M_3D_POSE_FILE = join(DATA_DIR, 'human3.6m', 'ground_truth_3d_pose.pkl')
HUMAN36M_KEYPOINT_DIR = join(DATA_DIR, 'human3.6m', 'cocopose')

NBA2K_MANIFEST_FILE = join(DATA_DIR, 'nba2k', 'pose_manifest.json')
NBA2K_3D_POSE_FILE = join(DATA_DIR, 'nba2k', 'ground_truth_3d_pose.pkl')
NBA2K_KEYPOINT_DIR = join(DATA_DIR, 'nba2k', 'cocopose')

AMASS_MANIFEST_FILE = join(DATA_DIR, 'amass', 'pose_manifest.json')
AMASS_3D_POSE_FILE = join(DATA_DIR, 'amass', 'ground_truth_3d_pose.pkl')
AMASS_KEYPOINT_DIR = join(DATA_DIR, 'amass', 'cocopose')
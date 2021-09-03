# Code for *Video Pose Distillation*

See our [project website](https://jhong93.github.io/projects/vpd.html) for the paper and details. To appear in ICCV 2021.

```
@inproceedings{vpd_iccv21,
    author={Hong, James and Fisher, Matthew and Gharbi, Micha\"{e}l and Fatahalian, Kayvon},
    title={{V}ideo {P}ose {D}istillation for {F}ew-{S}hot, {F}ine-{G}rained {S}ports {A}ction {R}ecognition},
    booktitle={ICCV},
    year={2021}
}
```

For code in this repository, see [LICENSE](LICENSE).

## Usage

This repository contains code for VPD and VIPE*, as described in our paper.

### VIPE*

To apply the VIPE* model:
```
./apply_vipe_model.py <pose_dir> <model_dir> -o <out_dir>
```
* pose_dir : the directory containing the 2D poses for each video
* model_dir : path to trained model
* out_dir : path to save features to

To train a VIPE* model see ```train_vipe_model.py```.
Example: ```./train_vipe_model.py --dataset 3d --save_dir <model_dir>```
Preprocessed 3D pose data for training is available here: [VIPE-data.zip](https://drive.google.com/drive/folders/1QuZ6tUNalGQeU9CpQf_ryZ4y4a7zfE_l?usp=sharing).
This archive includes ground truth 3D pose and 2D pose from different camera views.
Extract to ```data/vipe``` or update the paths in ```vipe_dataset_paths.py```.
For details on preprocessing, see [preprocess_3d_pose.py](preprocess_3d_pose.py).

A pre-trained VIPE model is available: [VIPE-model.zip](https://drive.google.com/drive/folders/1QuZ6tUNalGQeU9CpQf_ryZ4y4a7zfE_l?usp=sharing).

### VPD

#### Data preparation
To prepare the sports datasets, there are several steps:
1. Fetching the videos
2. Pose detection / tracking
3. Extracting crops (see [extract_square_crops.py](extract_square_crops.py))
4. Computing optical flow (see [raft/README.md](raft/README.md))

Our pose and tracking annotations can be found here: [URL](https://drive.google.com/drive/folders/1QuZ6tUNalGQeU9CpQf_ryZ4y4a7zfE_l?usp=sharing)

For the source videos:
* Diving48 : see original authors' website
* Floor exercise : obtain from FineGym authors, recut using [recut_finegym_video.py](recut_finegym_video.py). If using our pose annotations, make sure the frame rates match for each video or adapt accordingly.
* Figure skating : see [fs-videos.csv](https://drive.google.com/drive/folders/1QuZ6tUNalGQeU9CpQf_ryZ4y4a7zfE_l?usp=sharing) and [recut_fs_video.py](recut_fs_video.py)
* Tennis : see [tennis-videos.csv](https://drive.google.com/drive/folders/1QuZ6tUNalGQeU9CpQf_ryZ4y4a7zfE_l?usp=sharing)

It is recommended to unzip the files to the paths defined in ```video_dataset_paths.py``` or to update those paths to where the pose files are stored. For example:
```
diving48
|---pose
|---crops
\---videos
fs
|---pose
|---crops
\---videos
...
```

#### To train a student model:
```
./train_vpd_model.py <dataset> --save_dir <model_dir> --emb_dir <teacher_dir> --flow_img <flow_name> --motion
```
* dataset : the sports dataset to specialize to (e.g., ```fs```)
* model_dir : path to save models to
* flow_name : the name of the flow images for the crops, which have names ```<frame_no>.<flow_name>.png```
* teacher_dir : path to the teacher's features

#### To apply a student model:
```
./apply_vpd_model.py <model_dir> -d <dataset> -o <out_dir> --flow_img <flow_name>
```
* model_dir : path to the trained model
* out_dir : path to save features to
* flow_name : should be the same used for training

The student maintains the same output file formats as the teacher.

### Downstream tasks:

For action recognition:
```
./recognize.py -d <dataset> <feature_dir>
```
* dataset : the sports dataset
* feature_dir : the directory containing the pose features

See options such as ```--retrieve``` for the retrieval task.
For detection, see ```detect.py```.

Pre-trained VPD and VIPE* features/embeddings are available at [URL](https://drive.google.com/drive/folders/1QuZ6tUNalGQeU9CpQf_ryZ4y4a7zfE_l?usp=sharing).

## Data formats

### Video naming conventions

For Diving48 and FineGym, we maintain the original authors' video naming scheme.

For figure skating, videos (routines) are named by ```<video>_<number>_<start_frame>_<end_frame>.mp4```.

For tennis, videos (points) are named by: ```<video>_<start_frame>_<end_frame>.mp4```. Pose for each video is prefixed by ```front__``` or ```back__``` to denote the player.

### 2D pose format

Pose for each video is organized as follows:
```
men_olympic_short_program_2010_01_00011475_00015700
|---boxes.json
|---coco_keypoints.json.gz
|---mask.json.gz
\---meta.json
```

The format for ```boxes.json``` is:
```
[
    [frame_num, [x, y, w, h]], ...
]
```

The format ```coco_keypoints.json.gz``` is:
```
[
    [
        frame_num, [[score, [x, y, w, h], [[x, y, score] * 17]]], ...]
    ],
    ...
]
```

The format of ```mask.json.gz```:
```
[
    [
        frame_num, [[score, [x, y, w, h], base64_encoded_png], ...]
    ],
    ...
]
```

### Crop directories

Crops around the athlete, for training VPD, are extracted per video (see [extract_square_crops.py](extract_square_crops.py)):

```
men_olympic_short_program_2010_01_00011475_00015700
|---0.png           // <frame_num>.png
|---0.prev.png
|---0.flow.png
|---0.mask.png
|---1.png
|---1.prev.png
...
```

For tennis, the format is slightly different:
```
usopen_2015_mens_final_federer_djokovic
|---back
|   |---0.png       // <frame_num>.png
|   |---0.prev.png
|   |---0.flow.png
|   |---0.mask.png
|   ...
|
\---front
    |---0.png
    |---0.prev.png
    |---0.flow.png
    |---0.mask.png
    ...
```

### Features / embedding format

Embeddings are stored as pickle files, one per video. The format for each video
is:
```
[
    (frame_num, ndarray, {metadata dict}), ...
]
```
The ndarray may be 1D or 2D, depending on data augmentation (e.g., flip).
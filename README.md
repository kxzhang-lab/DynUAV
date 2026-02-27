# DynUAV
This repository contains the source code for the statistical analysis of the challenging characteristics of the DynUAV dataset.

# For Code
## Environmental requirements
All relevant dependencies and their versions can be found in: environment.yaml. To minimize potential issues, please ensure you use the following versions:
* matplotlib=3.10.0
* python=3.12.9
* numpy=2.0.1
* opencv-python=4.11.0.86
* pandas=2.2.3

## Dataset 
### Dataset Structure
```
DynUAV-I/
├── videos/  # Original video files (.mp4)
├── img_annos/
|  ├── train/
|  ├── val/
|  ├── test/
```
Each sequence folder under ```train/val/test``` follows the MOTChallenge-style format:
```
<sequence_name>/
├── img1/            # Extracted image frames
├── gt/     
|   └── gt.txt       # Ground-truth annotations
├── det/
|   └── det.txt      # Public detection results
└── seqinfo.ini      # Sequence metadata
```
### Annotation Format
DynUAV follows the standard MOTChallenge annotation format.

#### Ground Truth (```gt.txt```)
Each line corresponds to one object instance in one frame:
```
frame_id, object_id, x, y, width, height, conf, class, visibility, unused
```
* ```(x,y)```denotes the top-left coordinate of the bounding box.
* Bounding boxes are defined in pixel coordinates.
* The remaining fields follow MOTChallenge conventions.

#### Detection File (```det.txt```)
Detection entries follow the same bounding box format, with object_id = -1 and conf indicating detection confidence.

#### Sequence Metadata (```seqinfo.ini```)
Contains sequence-level information such as:
* Frame rate
* Sequence length
* Image resolution
  


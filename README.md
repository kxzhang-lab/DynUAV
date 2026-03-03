# DynUAV
This repository contains the source code for the statistical analysis of the challenging characteristics of the DynUAV dataset.

## For Code
### Environmental requirements
The statistical analysis scripts were tested under the following environment:
* matplotlib 3.10.0
* python 3.12.9
* numpy 2.0.1
* opencv-python 4.11.0.86
* pandas 2.2.3

These scripts are lightweight and do not require a dedicated conda environment. 
Using the above versions is recommended to ensure reproducibility.

### How to Run
The statistical analysis can be launched via:
#### Command-line Arguments
- `--dataset_name {UAV123, VisDrone, UAVDT, MDMT, MOT20, MOT17, DanceTrack, SportsMOT, DBT70, NAT2021, UAVTrack112, OURS}`
  Specifies which dataset to analyze. The default value is `OURS` (DynUAV).
- ```--show_process {True/False}```
  Whether to visualize intermediate results during analysis, including:
  - Sampled annotation visualization
  - Adjacent-frame non-overlap (IoU=0) visualization
  - Object trajectory visualization
Set ```--show_process False``` for faster execution without visualization.

#### Output Statistics
A. Video-level Statistics
 - Sequence length statistics
 - Bar chart of video lengths
 - Histogram of video length distribution

B. Annotation-level Statistics
 - Object density statistics
 - Distribution of object area ratio and aspect ratio
 - Distribution of area ratio and aspect ratio relative to the first appearance (reflecting viewpoint and scale variation)
 - Per-class sample count (each sample corresponds to one object instance per frame)
 - Number of trajectories per video
 - Number of samples per video
 - Distribution of sample counts

C. Trajectory-level Statistics
 - Trajectory interval statistics and distribution
 - Object lifetime statistics, bar chart, and histogram
 - Number of continuous trajectory segments per object
 - Adjacent-frame IoU distribution and statistics
 - Proportion of non-overlapping adjacent frames (IoU = 0)
 - Total displacement per object (sum of adjacent-frame displacements)
  


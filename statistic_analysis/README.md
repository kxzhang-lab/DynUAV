# DynUAV
This repository contains the source code for the statistical analysis of the challenging characteristics of the DynUAV dataset. [Click here to download the dataset (extract code: FFLU.).](https://pan.baidu.com/s/1ExzMaawct6Igwpi34aiDXQ?pwd=FFLU)

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

### Core Dataset Characteristics

#### Aggressive Maneuver-Induced Motion

DynUAV exhibits severe inter-frame motion caused by UAV maneuvers.
This is quantitatively reflected by:
 * Adjacent-frame IoU statistics
 * Proportion of non-overlapping adjacent frames (IoU = 0)
These metrics collectively characterize motion severity.

| Dataset    | Mean IoU(%)| IoU Variance(%) | Non-Overlap Ratio(%) |
|------------|------------|-----------------|----------------------|
| DynUAV     | 74.44    | 29.34        | 1.62             |
| MDMT       | 79.57    | 17.78        | 0.0818           |
| UAVDT      | 91.97    | 8.92        | 0.0329            |
| VisDrone   | 84.21    | 21.84        | 0.6375            |
| MOT17      | 93.83    | 12.15        | 0.0466            |
| MOT20      | 95.95    | 3.85        | 0            |
| DanceTrack | 89.05    | 9.34        | 0.001            |
| SportsMOT  | 77.72    | 16.89        | 0.0413           |

Table 1. Adjacent-Frame IoU Statistics and Non-Overlap Ratio.
Non-Overlap Ratio denotes the percentage of adjacent-frame object pairs with no overlap region.

#### Long Temporal Span
   
Objects in DynUAV often reappear after long temporal gaps.
This property is measured using:   
 * Trajectory interval statistics

| Dataset   | Mean Interval | Variance | Max Interval | Min Interval |
|------------|--------------|----------|--------------|--------------|
| DynUAV     | 142.22        | 228.40    | 1666           |  2          |
| MDMT     | 99.99        | 96.30    | 549            |  11          |
| UAVDT      | 99.87        | 125.16    | 512           |  6          |
| VisDrone   | 90.67        | 119.40    | 503           |  3          |
| MOT17      |  -        | -    | -           |  -          |
| MOT20      | -        | -    | -           |  -          |
| DanceTrack | 11.52        | 18.90    | 247           |  2          |
| SportsMOT  | 112.81       | 155.92   | 630          |  2           |

Table 2. Trajectory Interval Statistics
   
#### Frequent Trajectory Fragmentation
   
Trajectories are frequently interrupted due to dynamic viewpoint changes and occlusion.
This characteristic is captured by:
 * Number of continuous trajectory segments per object

| Dataset   | Mean Segments per ID | Variance(%) | Max Segments per ID | Min Segments per ID |
|------------|----------------------|----------|----------------------|----------------------|
| DynUAV     | 1.2391               | 68.80    |8                     |1                     |
| MDMT       | 1.0161               | 13.34    |3                     |1                     |
| UAVDT      | 1.03               | 19.39    | 4               | 1               |
| VisDrone   | 1.0437               | 23.70    | 9               | 1               |
| MOT17      | 1               | 0    | 1               | 1               |
| MOT20      | 1               | 0    | 1               | 1               |
| DanceTrack | 5.02               | 547.69   | 49               | 1               |
| SportsMOT  | 1.7380               | 1.0249    |5                   | 1                 |

Table 3. Continuous Trajectory Segment Statistics (Mean ± Variance)

The average number of continuous trajectory segments per object reflects how frequently identities are interrupted. 
A higher mean value indicates that objects are more frequently fragmented, posing greater re-identification difficulty.

The variance further characterizes the distribution spread of fragmentation levels. 
A larger variance suggests heterogeneous trajectory behaviors — 
some objects are frequently interrupted, while others remain relatively stable.
  


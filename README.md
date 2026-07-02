# DynUAV TrackTrack Baseline

This repository provides the TrackTrack baseline for **DynUAV**, a UAV-perspective multi-object tracking benchmark introduced in:

**Breaking Smooth-Motion Assumptions: A UAV Benchmark for Multi-Object Tracking in Complex and Adverse Conditions**, CVPR 2026.

- Paper: [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Breaking_Smooth-Motion_Assumptions_A_UAV_Benchmark_for_Multi-Object_Tracking_in_CVPR_2026_paper.pdf)
- arXiv: [2603.05970](https://arxiv.org/abs/2603.05970)
- Baseline tracker: [TrackTrack, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)

DynUAV contains 42 UAV videos, 37,893 frames, and more than 1.7M bounding-box annotations. It is designed to evaluate MOT under severe UAV ego-motion, drastic scale/viewpoint changes, long sequences, small objects, and motion blur.

## News

- 07/2026: We release the TrackTrack baseline code and the precomputed `dyn.pickle` feature file for DynUAV.
- 03/2026: DynUAV is accepted to CVPR 2026.

## Environment

The experiments were run in the `tracktrack` conda environment:

```bash
conda activate tracktrack
pip install -r requirements.txt
```

Our local environment:

```text
Python 3.8.20
PyTorch 2.0.0+cu118
OpenCV 4.12.0
NumPy 1.24.4
SciPy 1.10.1
```

Install PyTorch according to your own CUDA version.

## Dataset

Download DynUAV from:

- Baidu Netdisk: [DynUAV](https://pan.baidu.com/share/init?surl=ExzMaawct6Igwpi34aiDXQ&pwd=FFLU) (code: `FFLU`)
- Hugging Face: coming soon

Set the DynUAV dataset root:

```bash
export DYN_UAV_ROOT=/path/to/DynUAV
```

Expected test-set layout:

```text
{DYN_UAV_ROOT}/test
|-- 004
|   |-- img1
|   |-- gt
|-- 009
|-- 016
|-- 027
|-- 029
|-- 055
|-- 067
```

## Precomputed Features

To make reproduction simple, we provide:

```text
FastReID/dyn.pickle
```

This file contains the DynUAV detections and ReID embeddings used by our TrackTrack baseline. We generated it following the original TrackTrack-style detection and FastReID feature extraction pipeline; YOLOX and FastReID details are therefore not required for reproducing the reported tracking result.

Because `dyn.pickle` is large, it is intentionally excluded from Git. Please download it from [Google Drive](https://drive.google.com/file/d/1aOONn_-1u6tgc0n-VvVViiGWmg4nZ7bZ/view?usp=sharing) and place it at `FastReID/dyn.pickle`.

## Run

From the repository root:

With camera motion compensation (CMC):

```bash
cd Tracker
python run_ours.py \
  --data_dir "$DYN_UAV_ROOT/test" \
  --pickle_path "../FastReID/dyn.pickle" \
  --pickle_path_95 "../FastReID/dyn.pickle" \
  --output_dir "../outputs/3_track/dynuav_test_cmc" \
  --dataset "DynUAV" \
  --mode "test" \
  --vid_names "004,009,016,027,029,055,067"
cd ..
```

Without CMC:

```bash
cd Tracker
python run_ours.py \
  --data_dir "$DYN_UAV_ROOT/test" \
  --pickle_path "../FastReID/dyn.pickle" \
  --pickle_path_95 "../FastReID/dyn.pickle" \
  --output_dir "../outputs/3_track/dynuav_test_no_cmc" \
  --dataset "DynUAV" \
  --mode "test" \
  --vid_names "004,009,016,027,029,055,067" \
  --disable_cmc
cd ..
```

Results are saved to:

```text
outputs/3_track/dynuav_test_cmc/dyn/
|-- 004.txt
|-- 009.txt
|-- ...
```

CMC files for the DynUAV test sequences are provided in `Tracker/trackers/cmc/`. By default, `run_ours.py` loads these CMC files when available; `--disable_cmc` forces identity transforms.

## Result

The following TrackTrack results on the DynUAV test set are reported in the paper. All metrics except IDSW and IDs are percentages.

| Tracker | CMC | FP | FN | DetA | MOTA | HOTA | IDF1 | AssA | IDSW | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrackTrack | No | 15030 | 75740 | 56.30 | 65.31 | 60.64 | 71.04 | 65.95 | 439 | 1343 |
| TrackTrack | Yes | 15617 | 71034 | 57.65 | 66.95 | 62.74 | 74.81 | 68.89 | 256 | 1125 |

## Acknowledgement

This baseline is based on [TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html), with components from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FastReID](https://github.com/JDAI-CV/fast-reid), [TrackEval](https://github.com/JonathonLuiten/TrackEval), and [MOTChallenge](https://motchallenge.net/).

## License

The code is released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use DynUAV or this baseline, please cite:

```bibtex
@inproceedings{ye2026dynuav,
  title     = {Breaking Smooth-Motion Assumptions: A UAV Benchmark for Multi-Object Tracking in Complex and Adverse Conditions},
  author    = {Ye, Jingtao and Zhang, Kexin and Ma, Xunchi and Li, Yuehan and Zhu, Guangming and Shen, Peiyi and Jiang, Linhua and Zhang, Xiangdong and Zhang, Liang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}

@inproceedings{shim2025tracktrack,
  title     = {Focusing on Tracks for Online Multi-Object Tracking},
  author    = {Shim, Kyujin and Ko, Kangwook and Yang, Yujin and Kim, Changick},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {11687--11696},
  year      = {2025}
}
```

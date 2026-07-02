## Model Zoo
Save weights files under "./weights/"
  - [mot17_half_sbs_S50.pth](https://drive.google.com/file/d/1kTG7mVNhYGicR0IXZ0Y1rebVoBRfOMGY/view?usp=drive_link)
  - [mot17_sbs_S50.pth](https://drive.google.com/file/d/1rUYqWIj0nsQ23rDSv8NVx0Rrp3Lco1KP/view?usp=drive_link)
  - [mot20_half_sbs_S50.pth](https://drive.google.com/file/d/1xMI_PpfeY02yfkHzRHZfA4KZtRqHak1o/view?usp=drive_link)
  - [mot20_sbs_S50.pth](https://drive.google.com/file/d/1RhMnTt9JCuZUWk-jPhDPX2NQCZ5g_O3m/view?usp=drive_link)
  - [dance_sbs_S50.pth](https://drive.google.com/file/d/1c9Vn4PADNKFrCuS0HxhPz3PcTvvLWVhc/view?usp=drive_link)


## Training
Trained weights will be created under "./weights/"
```
# For MOT17
python train_net.py --num-gpus 1 --config-file 'configs/MOT17_half/sbs_S50.yml'
python train_net.py --num-gpus 1 --config-file 'configs/MOT17/sbs_S50.yml'

# For MOT20
python train_net.py --num-gpus 1 --config-file 'configs/MOT20_half/sbs_S50.yml'
python train_net.py --num-gpus 1 --config-file 'configs/MOT20/sbs_S50.yml'

# For DanceTrack
python train_net.py --num-gpus 1 --config-file 'configs/DanceTrack/sbs_S50.yml'
```


## Feature Extraction
Detection + feature extraction results will be created under "../outputs/2. det_feat/" as pickle files
```
# For MOT17 validation
python ext_feats.py --data_path '../../dataset/MOT17/train/' --pickle_path '../outputs/1. det/mot17_val_0.80.pickle' --output_path '../outputs/2. det_feat/mot17_val_0.80.pickle' --config_path 'configs/MOT17_half/sbs_S50.yml' --weight_path 'weights/mot17_half_sbs_S50.pth'
python ext_feats.py --data_path '../../dataset/MOT17/train/' --pickle_path '../outputs/1. det/mot17_val_0.95.pickle' --output_path '../outputs/2. det_feat/mot17_val_0.95.pickle' --config_path 'configs/MOT17_half/sbs_S50.yml' --weight_path 'weights/mot17_half_sbs_S50.pth'

# For MOT17 test
python ext_feats.py --data_path '../../dataset/MOT17/test/' --pickle_path '../outputs/1. det/mot17_test_0.80.pickle' --output_path '../outputs/2. det_feat/mot17_test_0.80.pickle' --config_path 'configs/MOT17/sbs_S50.yml' --weight_path 'weights/mot17_sbs_S50.pth'
python ext_feats.py --data_path '../../dataset/MOT17/test/' --pickle_path '../outputs/1. det/mot17_test_0.95.pickle' --output_path '../outputs/2. det_feat/mot17_test_0.95.pickle' --config_path 'configs/MOT17/sbs_S50.yml' --weight_path 'weights/mot17_sbs_S50.pth'

# For MOT20 validation
python ext_feats.py --data_path '../../dataset/MOT20/train/' --pickle_path '../outputs/1. det/mot20_val_0.80.pickle' --output_path '../outputs/2. det_feat/mot20_val_0.80.pickle' --config_path 'configs/MOT20_half/sbs_S50.yml' --weight_path 'weights/mot20_half_sbs_S50.pth'
python ext_feats.py --data_path '../../dataset/MOT20/train/' --pickle_path '../outputs/1. det/mot20_val_0.95.pickle' --output_path '../outputs/2. det_feat/mot20_val_0.95.pickle' --config_path 'configs/MOT20_half/sbs_S50.yml' --weight_path 'weights/mot20_half_sbs_S50.pth'

# For MOT20 test
python ext_feats.py --data_path '../../dataset/MOT20/test/' --pickle_path '../outputs/1. det/mot20_test_0.80.pickle' --output_path '../outputs/2. det_feat/mot20_test_0.80.pickle' --config_path 'configs/MOT20/sbs_S50.yml' --weight_path 'weights/mot20_sbs_S50.pth'
python ext_feats.py --data_path '../../dataset/MOT20/test/' --pickle_path '../outputs/1. det/mot20_test_0.95.pickle' --output_path '../outputs/2. det_feat/mot20_test_0.95.pickle' --config_path 'configs/MOT20/sbs_S50.yml' --weight_path 'weights/mot20_sbs_S50.pth'

# For DanceTrack validation
python ext_feats.py --data_path '../../dataset/DanceTrack/val/' --pickle_path '../outputs/1. det/dance_val_0.80.pickle' --output_path '../outputs/2. det_feat/dance_val_0.80.pickle' --config_path 'configs/DanceTrack/sbs_S50.yml' --weight_path 'weights/dance_sbs_S50.pth'
python ext_feats.py --data_path '../../dataset/DanceTrack/val/' --pickle_path '../outputs/1. det/dance_val_0.95.pickle' --output_path '../outputs/2. det_feat/dance_val_0.95.pickle' --config_path 'configs/DanceTrack/sbs_S50.yml' --weight_path 'weights/dance_sbs_S50.pth'

# For DanceTrack test
python ext_feats.py --data_path '../../dataset/DanceTrack/test/' --pickle_path '../outputs/1. det/dance_test_0.80.pickle' --output_path '../outputs/2. det_feat/dance_test_0.80.pickle' --config_path 'configs/DanceTrack/sbs_S50.yml' --weight_path 'weights/dance_sbs_S50.pth'
python ext_feats.py --data_path '../../dataset/DanceTrack/test/' --pickle_path '../outputs/1. det/dance_test_0.95.pickle' --output_path '../outputs/2. det_feat/dance_test_0.95.pickle' --config_path 'configs/DanceTrack/sbs_S50.yml' --weight_path 'weights/dance_sbs_S50.pth'

```


## Reference
  - https://github.com/JDAI-CV/fast-reid

python ext_feats_custom.py \
    --data_path /home/ye-jingtao/uav_dataset/ours/DHUAV \
    --pickle_path /home/ye-jingtao/uav_dataset/ours/DHUAV/dhuav_det.pkl \
    --output_path /home/ye-jingtao/uav_dataset/ours/DHUAV/dhuav_reid.pkl \
    --config_path 'configs/MOT17/sbs_S50.yml' \
    --weight_path 'weights/mot17_sbs_S50.pth'


python run_ours.py \
    --data_dir /home/ye-jingtao/uav_dataset/ours/DHUAV \
    --pickle_path /home/ye-jingtao/uav_dataset/ours/DHUAV/dhuav_reid.pickle \
    --pickle_path_95 /home/ye-jingtao/uav_dataset/ours/DHUAV/dhuav_reid.pickle \
    --output_dir /home/ye-jingtao/uav_mot/TrackTrack-main/Tracker/outputs \
    --dataset "DHUAV" \
    --mode "test"

python run_ours.py \
    --data_dir /home/ye-jingtao/uav_dataset/ours/test \
    --pickle_path /home/ye-jingtao/uav_mot/TrackTrack-main/FastReID/val.pickle \
    --pickle_path_95 /home/ye-jingtao/uav_mot/TrackTrack-main/FastReID/val.pickle \
    --output_dir /home/ye-jingtao/uav_mot/TrackTrack-main/Tracker/outputs/with_cmc/0702 \
    --dataset "UAV_Custom" \
    --mode "test"
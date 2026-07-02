## Run
Tracking results will be created under "../outputs/3. track/"

```
# For MOT17 validation
python run.py --dataset "MOT17" --mode "val"

# For MOT17 test
python run.py --dataset "MOT17" --mode "test"
python gen_test_file.py

# For MOT20 validation
python run.py --dataset "MOT20" --mode "val"

# For MOT20 test
python run.py --dataset "MOT20" --mode "test"
python gen_test_file.py

# For DanceTrack validation
python run.py --dataset "DanceTrack" --mode "val"

# For DanceTrack test
python run.py --dataset "DanceTrack" --mode "test"
```

# Multi-Modal

We use only IR images for multi-modal training and testing

## Requirements

- pytorch 1.4

## File structure

- We use a face detector to detect face rois with RGB files and mapping them to IR/Depth image files
- Context in annotation file: image_name coord_x1 coord_y1 coord_x2 coord_y2 image_label
- Annotaion files are all in anno dir, for example: anno/IR/4@1_train.txt
- Train/test config file are all in configs dir, for example: configs/config_20200301_4@1_ir_01.py
- config file sets up train/test settings
- change the settings with `conf.data` to `$YOUR_DATA_PATH` and `conf.model.save_path` to `$YOUR_DUMP_PATH`
- test result will save to `conf.test.pred_path`, change it to `$YOUR_PATH`

## Train/Test

execute commands below for training 4@1/4@2/4@3:

- python3 train.py configs/config_20200301_4@1_ir_01.py
- python3 train.py configs/config_20200301_4@2_ir_02.py
- python3 train.py configs/config_20200301_4@3_ir_02.py

execute commands below for testing:

- python3 test.py configs/config_20200301_4@1_ir_01.py
- python3 test.py configs/config_20200301_4@2_ir_02.py
- python3 test.py configs/config_20200301_4@3_ir_02.py

Use `tools/process.py` to convert results of still images to videos and get final results:

- python3 tools/process.py `$YOUR_RESULT_DIR_OF_4@1` > 4@1_result.txt
- python3 tools/process.py `$YOUR_RESULT_DIR_OF_4@2` > 4@2_result.txt
- python3 tools/process.py `$YOUR_RESULT_DIR_OF_4@3` > 4@3_result.txt

Plz check your final results in txt files above, like `4@1/2/3_result.txt`

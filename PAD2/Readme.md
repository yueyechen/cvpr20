# Multi-Modal

We use only IR images for multi-modal training and testing

## Requirements

- pytorch 1.4

## File structure

- We use a face detector to detect face rois with RGB files
- And mapping them to IR/Depth image files
- Annotaion files are all in anno dir
- Train/test config file are all in configs dir

## Train/Test

execute commands below for training:

```python3 train.py configs/config_xxxxx.py```

execute commands below for testing:

```python3 test.py configs/config_xxxxx.py```

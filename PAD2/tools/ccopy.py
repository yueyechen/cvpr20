import os
import sys

from shutil import copy, copytree

fname = sys.argv[1]
src_dir = sys.argv[2]
dst_dir = sys.argv[3]

lines = open(fname).readlines()
for l in lines:
    path = l.strip().split()[0]
    src_path = os.path.join(src_dir, path)
    dst_path = os.path.join(dst_dir, path)
    if os.path.isdir(src_path):
        copytree(src_path, dst_path)
    else:
        save_dir = os.path.dirname(dst_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        copy(src_path, dst_path)
        print(dst_path)


import os
import sys

from shutil import copytree

fname = sys.argv[1]
src_dir = sys.argv[2]
dst_dir = sys.argv[3]

lines = open(fname).readlines()
for l in lines:
    src_path = os.path.join(src_dir, l.strip())
    dst_path = os.path.join(dst_dir, l.strip())
    copytree(src_path, dst_path)


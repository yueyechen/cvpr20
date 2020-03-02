import os
import sys


fname = sys.argv[1]
dname = sys.argv[2]

lines = open(fname).readlines()

for l in lines:
    iid = l.strip().split('/')[-1]
    files = sorted(os.listdir(os.path.join(dname, iid, 'profile')))
    if int(iid) % 2 == 0:
        label = 1
    else:
        label = 0

    for f in files:
        out_str = '{} {}'.format(os.path.join(l.strip(), 'profile', f), label)
        print(out_str)


import os

path = '/tmp/tao.cai/jiachen.xue/0301/4@2_dev_03.txt'
with open(path, 'r') as f:
    lines = [x.strip() for x in f.readlines()]
    with open('../../2_dev.txt', 'w') as f:
        for line in lines:
            img_path = line.split()[0]
            liveness = line.split()[-1]
            write_str = '%s %s' %(img_path, liveness)
            f.write(write_str)
            f.write('\n')
    
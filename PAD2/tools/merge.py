import sys

roi_file = sys.argv[1]
clss_file = sys.argv[2]

dataset = {}
lines = open(roi_file).readlines()
for l in lines:
    path, *roi, score = l.strip().split()
    name = '/'.join(path.split('/')[-4:])
    if name in dataset:
        raise KeyError
    dataset[name] = roi

lines = open(clss_file).readlines()
for l in lines:
    name, label = l.strip().split()
    if name in dataset:
        out_str = '{} {} {}'.format(name, ' '.join(dataset[name]), label)
    else:
        out_str = '{} {} {}'.format(name, ' '.join(['-1'] * 4), label)
    print(out_str)


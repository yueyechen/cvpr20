import os
import sys


def process_file(in_file, label_file, save_file):
    with open(in_file, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    
    label_dict = {}
    with open(label_file, 'r') as f:
        label_inf = f.readlines()
        for val in label_inf:
            label_dict[val.split()[0]] = float(val.split()[-1])

    with open(save_file, 'w') as f:
        for line in lines:
            data = line.strip().split(' ')
            if len(data) != 2:
                print('wrong')
                continue
            img_path = data[0]
            # label = float(data[-3])
            liveness = float(data[-1])
            track_id = '/'.join(img_path.split('/')[:-2])
            try:
                label = label_dict[track_id]
            except:
                print(track_id)
                print(img_path)
                print('wrong')

            write_str = img_path + ' '  + str(liveness) + ' ' + str(label)
            f.write(write_str)
            f.write('\n')


if __name__ == "__main__":
    in_file = sys.argv[1]
    label_file = sys.argv[2]
    save_file = sys.argv[3]
    process_file(in_file, label_file, save_file)

               

import os
import cv2
import numpy as np
import sys
import copy


path = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'
save_root = '/mnt/cephfs/smartauto/users/guoli.wang/tao.cai/CycleGAN/input/test/trainB'

def select_image(input_file, expand_ratio=1.0):
    with open(input_file, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    image_count = len(lines)
    index = np.linspace(0, 130000, 600).astype('int32')
    for val in index:
        data_dict = {}
        data = lines[val].split()                
        img_path = data[0]
        label = int(data[-1])
        try:
            rects = [float(x) for x in data[-5:-1]]
        except:
            continue
        data_dict['img_path'] = os.path.join(path, img_path)
        data_dict['rects'] = rects
        img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), -1)
        if img is None:
            assert False, 'image `{}` is empty.'.format(data_dict['img_path'])
        rects = copy.deepcopy(data_dict['rects'])
        if rects == [-1.0] * 4:
        # print('not need to crop')
        # print(data_dict['img_path'])
            img_final = cv2.resize(img, (256, 256))
            save_path = os.path.join(save_root, '%d.jpg' %(val,))
            # img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img_final = Image.fromarray(img_final.astype('uint8'))
            cv2.imwrite(save_path, img_final)
            continue
        else:
            img_h, img_w = img.shape[:2]

            w = rects[2] - rects[0]
            h = rects[3] - rects[1]
            if h > w:
                origin = rects[0] + rects[2]
                rects[0] = np.maximum((origin / 2. - h / 2.), 0)
                rects[2] = np.minimum((origin / 2. + h / 2.), img_w)
            else:
                origin = rects[1] + rects[3]
                rects[1] = np.maximum((origin / 2. - w / 2.), 0)
                rects[3] = np.minimum((origin / 2. + w / 2.), img_h)

            w = rects[2] - rects[0]
            h = rects[3] - rects[1]
            expand_ratio = expand_ratio
            bbox1 = [np.maximum(rects[0] - w * (expand_ratio - 1) / 2., 0),
                    np.maximum(rects[1] - h * (expand_ratio - 1) / 2., 0),
                    np.minimum(rects[2] + w * (expand_ratio - 1) / 2., img_w),
                    np.minimum(rects[3] + h * (expand_ratio - 1) / 2., img_h)]
        
            bbox1 = [int(x) for x in bbox1]
            img_final = img[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
            img_final = cv2.resize(img_final, (256, 256))
            save_path = os.path.join(save_root, '%d.jpg' %(val,))
            cv2.imwrite(save_path, img_final)
            continue


if __name__ == '__main__':
    input_file = sys.argv[1]
    expand_ratio = 1.0
    select_image(input_file, expand_ratio=expand_ratio)
    print('finish')

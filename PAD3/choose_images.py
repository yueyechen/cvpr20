import os

path = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'
path = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase2'

id_path = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@3_dev_res.txt'
id_path = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase2/4@2_test_res.txt'
id_path = '/home/users/tao.cai/PAD/test_file/lstA.txt'

save_dir = '/home/users/tao.cai/competition_depth_images/'
#save_dir = '/home/users/tao.cai/competition_depth_images/phase2/list/'
save_dir = '/home/users/tao.cai/competition_profile_images/phase2/list/'
os.makedirs(save_dir)

with open(id_path, 'r') as f:
    id_list = [x.strip() for x in f.readlines()]


for val in id_list:
    image_path = os.path.join(path, val, 'profile/0001.jpg')
    save_path = os.path.join(save_dir, val.split('/')[-1] + '_0001.jpg')
    
    command = 'cp %s %s' %(image_path, save_path)
    #print(command)
    os.system(command)


# dev_path = '/home/users/tao.cai/4@3_dev_res.txt'
# save_dev_path = '/home/users/tao.cai/4@3_dev_res_label.txt'
# with open(dev_path, 'r') as f:
#     lines = [x.strip() for x in f.readlines()]
# count = 0
# with open(save_dev_path, 'w') as f:
#     for line in lines:
#         if count % 2 == 0:
#             f.write(line + ' 1')
#             f.write('\n')
#             count += 1
#         else:
#             f.write(line + ' 0')
#             f.write('\n')
#             count += 1
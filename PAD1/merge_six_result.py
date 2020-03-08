# coding: utf-8



def main():
    listpath_list = []
    listpath_list +=[
        '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result/res50_rgb_06_reg00_rect00_4@1/4@1_dev_e19_04.txt',
        '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result/res50_rgb_06_reg00_rect00_4@1/4@1_test_e19_04.txt',
        '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result/res101_rgb_06_reg00_rect00_4@2/4@2_dev_e15_04.txt',
        '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result/res101_rgb_06_reg00_rect00_4@2/4@2_test_e15_04.txt',
        '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result/res101_rgb_06_reg00_rect00_4@3/4@3_dev_e16_02.txt',
        '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result/res101_rgb_06_reg00_rect00_4@3/4@3_test_e16_02.txt',
    ]
    savelist = '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result/submit/0301.txt'

    fw = open(savelist, 'w')
    for listpath in listpath_list:
        with open(listpath, 'r')as fr:
            for line in fr.readlines():
                data = line.strip().split()
                name = data[0]
                pred = data[-1]
                write_str = name+' '+pred+'\n'
                fw.write(write_str)
    fw.close()


if __name__ == '__main__':
    main()
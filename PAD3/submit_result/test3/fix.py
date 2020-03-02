path = './0206_res18_00_result_method1_01_avg_w_label.txt'
with open(path, 'r') as f:
    lines = [x.strip() for x in f.readlines()]
with open('./1.txt', 'w') as f:
    for line in lines:
        data = line.split()
        write_str = '%s %s' %(data[0], data[-1]) + '\n'
        f.write(write_str)

# ReadME

## 1.数据集
本项目的数据来源于 Chalearn CASIA-CeFA[1]

## 2.环境配置
### 2.1 环境描述

本项目运行所使用系统为CentOS 7.6.1810，编程语言为python3，pytorch 1.2；
硬件环境为：
> Nvidia GPU支持（GPU：GeForce GTX 1080Ti）


## 3. 训练模型

**Step1**: 修改配置文件

对 config.py 中路径相关变量进行修改：
```python
conf.work_path = '<改成自己的work_path路径>'
conf.log_path = '<保存log的路径>', 默认是在文件夹 conf.work_path/log
conf.save_path = '<保存模型的路径>', 默认是在文件夹 conf.work_path/save
```

如果训练模型4@1，使用resnet50网络结构，对train.py 的变量进行以下修改：

```python
exp_sequence = ['4@1']
conf.data_path = '<训练数据所在的路径>' 
conf.train_list = '<anno/{}.txt的路径>', #处理好的list都在anno文件下
conf.val_list = '<anno/{}.txt的路径>'
conf.exp = '可以自己设置exp的name'
```
对 Learner.py 文件进行修改：
```python
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
```

如果训练模型4@2和4@3，使用resnet101网络结构，对train.py的变量进行以下修改：

```python
exp_sequence = ['4@2', '4@3']
conf.batch_size = 64
conf.data_path = '<训练数据所在的路径>' 
conf.train_list = '<anno/{}.txt的路径>', #处理好的list都在anno文件下
conf.val_list = '<anno/{}.txt的路径>'
conf.exp = '可以自己设置exp的name'
```
对 Learner.py 文件进行修改：
```python
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
```

**Step2**: 训练模型

运行 `train.py`文件

```
python3 train.py # 训练脚本
```



程序运行完毕以后，会在`conf.save_path/conf.exp`目录下生成训练模型.

## 4. 测试

**Step1**:  相关配置参数，保持与训练参数一致。

如果测试4@1模型，对 Learner.py 的修改是:
```python
os.environ['CUDA_VISIBLE_DEVICES']='0' #测试时候设置成单卡
self.model = resnet_face50(use_se=True) #4@1用的网络结构是resnet50
```
对test_reg_several_epoch.py的修改是：
```python
exp_sequence = ['4@1']
epoch_range = range(15, 20) #需要测试的epoch
conf.data_path = '<训练数据所在的路径>'
conf.result_path = './work_space/result' #保存结果的目录 # 指定生成预测文件的根目录，默认在当前工程目录的 `work_space/test_pred` 目录下
conf.exp = '' # 指定测试的exp的name
conf.model_path = '' # 指定保存模型的路径
conf.result_name = '' # 生成的txt的name
```
如果测试4@2和4@3模型，对 Learner.py 的修改是:
```python
os.environ['CUDA_VISIBLE_DEVICES']='0' #测试时候设置成单卡
self.model = resnet_face101(use_se=True) #4@2和4@3用的网络结构是resnet101
```
对test_reg_several_epoch.py的修改是：
```python
exp_sequence = ['4@2', '4@3']
epoch_range = range(15, 20) #需要测试的epoch
conf.data_path = '<训练数据所在的路径>'
conf.result_path = './work_space/result' #保存结果的目录 # 指定生成预测文件的根目录，默认在当前工程目录的 `work_space/test_pred` 目录下
conf.exp = '' # 指定测试的exp的name
conf.model_path = '' # 指定保存模型的路径
conf.result_name = '' # 生成的txt的name
```

**Step2**： 运行测试脚本
为了方便测试，提供我们训好的模型，[百度云盘链接](链接: https://pan.baidu.com/s/1FuTTeFwpBxxicB_Bttp__g) 提取码: pnat，可以复现我们提交的结果。
  
打开终端，切换到该项目下，执行以下命令：
```
python3 test_reg_several_epoch.py
```

**Step3**： 后处理
我们提供了两种后处理方式，4@1和4@2用post_process_method_04.py，4@3用post_process_method_02.py。只需要在python脚本中设置好对应的参数即可得到后处理结果。

处理4@1，需要修改post_process_method_04.py中：
```python
exp_seq = ['4@1']
epoch_range = range(15, 20) # 可以与test中的epoch_range对应，表示需要处理的epoch范围
list_type_seq = ['dev_frame', 'test_frame'] # 与test中生成的txt名字对应
exp = 'res50' # 最好保持和之前exp name对应
work_space = '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space' #同训练的work space一致
exp_name = '{}_{}_06_reg00_rect00_{}'.format(exp, format, exp_id) # 与训练、测试时候的形式一致
处理后的txt结果会保存在 os.path.join(work_space, 'result', exp_name, '{}_dev_e{}_04.txt'.format(exp_id,epoch)) 和 os.path.join(work_space, 'result', exp_name, '{}_test_e{}_04.txt'.format(exp_id,epoch))
```

处理4@2，需要修改post_process_method_04.py中：
```python
exp_seq = ['4@2']
epoch_range = range(15, 20) # 可以与test中的epoch_range对应，表示需要处理的epoch范围
list_type_seq = ['dev_frame', 'test_frame'] # 与test中生成的txt名字对应
exp = 'res101' # 最好保持和之前exp name对应
work_space = '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space' #同训练的work space一致
exp_name = '{}_{}_06_reg00_rect00_{}'.format(exp, format, exp_id) # 与训练、测试时候的形式一致
处理后的txt结果会保存在 os.path.join(work_space, 'result', exp_name, '{}_dev_e{}_04.txt'.format(exp_id,epoch)) 和 os.path.join(work_space, 'result', exp_name, '{}_test_e{}_04.txt'.format(exp_id,epoch))
```

处理4@3，需要修改post_process_method_02.py中：
```python
exp_seq = ['4@3']
epoch_range = range(15, 20) # 可以与test中的epoch_range对应，表示需要处理的epoch范围
list_type_seq = ['dev_frame', 'test_frame'] # 与test中生成的txt名字对应
exp = 'res101' # 最好保持和之前exp name对应
work_space = '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space' #同训练的work space一致
exp_name = '{}_{}_06_reg00_rect00_{}'.format(exp, format, exp_id) # 与训练、测试时候的形式一致
处理后的txt结果会保存在 os.path.join(work_space, 'result', exp_name, '{}_dev_e{}_04.txt'.format(exp_id,epoch)) 和 os.path.join(work_space, 'result', exp_name, '{}_test_e{}_04.txt'.format(exp_id,epoch))
```
在分别得到三组处理后的dev和test的txt后，就可以使用merge_six_result.py merge到一起了。


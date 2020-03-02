# ReadME

## 1.数据集
本项目的数据来源于 Chalearn CASIA-SURF[1](https://competitions.codalab.org/competitions/20853#learn_the_details) 

对CASIA数据集进行基本的预处理，并将数据上传至百度云盘[下载地址](https://pan.baidu.com/s/1_caiA9r9SPjRI09fNKsscg);提取码：9ooz 

百度云盘包括训练集、测试集以及训练集的list、测试集的list，请自行下载解压到 conf.data_folder 指定的目录下，并请确保conf.train_list，conf.test_list 的路径与实际放置的路径一致。

## 2.环境配置
### 2.1 环境描述

本项目运行所使用系统为ubuntu 16.04 LST，编程语言为python3，pytorch 0.41；
硬件环境为：
> Nvidia GPU支持（GPU：GeForce GTX 1080Ti）
> 
> CPU型号是 `Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz`

### 2.2 软件环境
该项目的requirements.txt中包含项目运行所必须的python 库。
打开终端，跳转到该项目下，执行以下命令，安装所需的python类库。
```
pip3 install -r requirements.txt
```



## 3. 训练模型

**Step1**: 修改配置文件

对config.py 的变量进行以下修改：

```python
conf.data_folder = '<训练数据所在的路径>' 
```

``` 
conf.train_list = '<train_list.txt的路径>'
```

**Step2**: 训练模型

运行 `train.py`文件

```
python3 train.py # 训练脚本
```



程序运行完毕以后，会在`conf.save_path/conf.exp`目录下生成训练模型.

## 4. 测试

**Step1**:  修改配置文件

对config.py 的变量进行以下修改：

```python

conf.save_path = './work_space/test_pred' # 指定生成预测文件的根目录，默认在当前工程目录的 `work_space/test_pred` 目录下
conf.exp = '' # 指定生成预测文件所在的目录

```

**Step2**： 运行测试脚本
为了方便测试，可以从百度云下载已经训练好的模型; [模型下载地址](https://pan.baidu.com/s/1c2KmizAjfduiuqw2xpCb6A ) ，提取码：ah6x 
  为了快速复现我们的实验结果，可以直接将模型改名为`epoch=1.pth`,并且放置到`conf.save_path/conf.exp/.`路径下，然后设置config.py中的相关参数，包括：
```
conf.test.epoch_start=1
conf.test.epoch_end=2
conf.test.interval=1
```
请确保conf.data_folder和conf.test_list设置正确。
打开终端，切换到该项目下，执行以下命令：
```
python3 test.py
```

运行以下命令进行测试，该脚本会针对每个epoch，从目录`${conf.save_path}/${conf.exp}`中读取模型并在目录`${conf.test.pred_path}/${conf.exp}`生成以 `epoch={%d}.txt` 命名的文件，每个 txt 文件保存了在测试集上的预测结果。


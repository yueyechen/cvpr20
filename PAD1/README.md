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

对train.py 的变量进行以下修改：

```python
conf.data_path = '<训练数据所在的路径>' 
```

``` 
conf.train_list = '<anno/train_list.txt的路径>'
```

``` 
conf.val_list = '<anno/val_list.txt的路径>'
```

**Step2**: 训练模型

运行 `train.py`文件

```
python3 train.py # 训练脚本
```



程序运行完毕以后，会在`conf.save_path/conf.exp`目录下生成训练模型.

## 4. 测试

**Step1**:  修改test_reg_several_epoch.py中的相关配置参数，保持与训练参数一致。


```python

conf.data_path = '<训练数据所在的路径>'
conf.result_path = './work_space/result'保存结果的根目录 # 指定生成预测文件的根目录，默认在当前工程目录的 `work_space/test_pred` 目录下
conf.exp = '' # 指定测试的exp版本
conf.model_path = '' # 指定保存模型的路径

```

**Step2**： 运行测试脚本
为了方便测试，提供我们训好的模型，[百度云盘链接](链接: https://pan.baidu.com/s/1FuTTeFwpBxxicB_Bttp__g) 提取码: pnat，可以复现我们提交的结果。
  
打开终端，切换到该项目下，执行以下命令：
```
python3 test_reg_several_epoch.py
```

**Step3**： 后处理
我们提供了两种后处理方式，4@1和4@2用post_process_method_04.py，4@3用post_process_method_02.py。只需要在python脚本中设置好对应的参数即可得到后处理结果。



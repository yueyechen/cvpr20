import torch.nn as nn
import torch


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding=0, dilation=1, groups=1, is_bn_relu=False, use_bias=False, name=" "):
        super(ConvBnRelu, self).__init__()

        self.block_name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.is_bn_relu = is_bn_relu
        self.relu = nn.ReLU(inplace=True)

        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, dilation=dilation,
            padding=padding, groups=groups, use_bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def hybrid_forward(self, x):
        print("********************************")
        print("this is for checking ConvBnRelu Block Parameters")
        print("%s --- in_channels: %d, out_channels: %d, kernel_size: %d, strides: %d, padding: %d, groups: %d, is_bn_relu: %s"
            % (self.block_name, self.in_channels, self.out_channels, self.kernel, self.strides, self.padding, self.groups, self.is_bn_relu))

        x = self.conv(x)
        if self.is_bn_relu:
            x = self.relu(self.bn(x))
            print("Conv BN ReLU")
        else:
            x = self.bn(nn.relu(x))
            print("Conv ReLU BN")
        return x


class Stem_Block(nn.Module):
    '''
    Stem Block is placed before stage and after input image
    '''
    def __init__(self, in_channels, out_channels, image_size, use_se=False, **kwargs):
        super(Stem_Block, self).__init__(**kwargs)
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channel=out_channels, kernel_size=3, stride=2, padding=1, is_bn_relu=True)

        self.branch_maxpool = nn.MaxPool2d((2, 2), (2, 2))
        self.branch_conv2 = ConvBnRelu(in_channels=out_channels, out_channel=out_channels//2, kernel_size=1, stride=1, padding=0, is_bn_relu=True)

        self.branch_conv3 = ConvBnRelu(in_channels=out_channels//2, out_channel=out_channels, kernel_size=3, stride=2, padding=1, is_bn_relu=True)


        self.concat_conv4 = ConvBnRelu(in_channels=out_channels*2, out_channel=out_channels, kernel_size=1, stride=1, padding=0, is_bn_relu=True)


    def hybrid_forward(self, x):
        x = self.conv1(x)

        # branch 1
        x_branch1 = self.branch_conv2(x)
        x_branch1 = self.branch_conv3(x_branch1)

        # branch 2
        x_branch2 = self.branch_maxpool(x)

        # concat & conv
        x_concat = torch.cat(x_branch1, x_branch2, dim = 1)
        x_out = self.concat_conv4(x_concat)

        print("#------------------------#")
        print("this is for VarGNet explore stem block")
        print("feature maps: %d, in_channels: %d, out_channels: %d" % (self.image_size, self.in_channels, self.out_channels))

        return x_out


class VarGNet_explore_v2_UnitB(nn.Module):

    # """VarGNet v1 unit for stride=2"""
    # Variable Group Network, Transition Block, Highlight:
    # 1: Inverted Residual Structure Inspired by MobileNet-V2
    # 2: Merge two Depthwise Seperable Conv together
    # 3: Cardinality Introduced By ResNeXt

    def __init__(self, in_channels, out_channels, image_size, strides=2, group_base=8, factor=2, use_se=False, name=" ", **kwargs):
        super(VarGNet_explore_v2_UnitB, self).__init__(**kwargs)

        if strides==2:
            assert out_channels // in_channels == 2

        assert in_channels % 8 == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_increased_channels = in_channels * factor
        self.second_increased_channels = self.out_channels * factor
        self.group_base = group_base
        self.feature_map = image_size
        self.relu = nn.ReLU(inplace=True)

        
        
        self.se = None

            ## First cardinality Inverted Residual Structure
            ## First group conv takes place of depthwise conv
        self.group_conv1 = ConvBnRelu(in_channels=self.in_channels, out_channels=self.first_increased_channels, kernel_size=3, strides=strides, padding=1, 
            groups=self.in_channels // self.group_base, is_bn_relu=True, name=name+"group_conv1")

            ## First pointwise conv 
        self.pointwise_conv2 = ConvBnRelu(in_channels=self.first_increased_channels, out_channels=self.out_channels, 
            kernel_size=1, strides=1, padding=0, is_bn_relu=True, name=name+"pointwise_conv2")


            ## Second cardinality Inverted Residual Structure
            ## Second group conv takes place of depthwise conv
        self.group_conv3 = ConvBnRelu(in_channels=self.in_channels , out_channels=self.first_increased_channels, kernel_size=3, strides=strides, padding=1, 
            groups=self.in_channels // self.group_base, is_bn_relu=True, name=name+"group_conv3")

            ## Second pointwise conv
        self.pointwise_conv4 = ConvBnRelu(in_channels=self.first_increased_channels, out_channels=self.out_channels, 
            kernel_size=1, strides=1, padding=0, is_bn_relu=True, name=name+"pointwise_conv4")


            ## group conv takes place of depthwise conv after merging two cardinality together
        self.group_conv5 = ConvBnRelu(in_channels=self.out_channels, out_channels=self.second_increased_channels, 
            kernel_size=3, strides=1, padding=1, groups=self.first_increased_channels // self.group_base, is_bn_relu=True, 
            name=name+"group_conv5")

            ## pointwise conv after merging two cardinality together
        self.pointwise_conv6 = ConvBnRelu(in_channels=self.second_increased_channels, out_channels=out_channels, 
            kernel_size=1, strides=1, padding=0, is_bn_relu=True, name=name+"pointwise_conv6")


            ## Short Cut Conv
            ## group conv takes place of depthwise conv
        self.group_conv7 = ConvBnRelu(in_channels=self.in_channels, out_channels=self.first_increased_channels, kernel_size=3, strides=strides, padding=1, 
            groups=self.in_channels // self.group_base, is_bn_relu=True, name=name+"group_conv7")

            ## pointwise conv 
        self.pointwise_conv8 = ConvBnRelu(in_channels=self.first_increased_channels, out_channels=out_channels, 
            kernel_size=1, strides=1, padding=0, is_bn_relu=True, name=name+"pointwise_conv8")


    def hybrid_forward(self, x):
        # Cardinality Inverted Residual Structure
        print("###########################################################################################")
        print("VarGNet Transition Block, Input Feature Size: %d x %d" % (self.feature_map, self.feature_map))
        print("VarGNet Transition Block, Input Channels: %d" % (self.in_channels))
        print("VarGNet Transition Block, Group Base Channels: %d, Group Numbers: %d" % (self.group_base, self.in_channels // self.group_base))

        print("#------------------------------------------------------------")
        x_cardi1 = self.group_conv1(x)
        x_cardi1 = self.pointwise_conv2(x_cardi1)

        x_cardi2 = self.group_conv3(x)
        x_cardi2 = self.pointwise_conv4(x_cardi2)

        x_cardi = self.relu(x_cardi1 + x_cardi2)


        x_cardi = self.group_conv5(x_cardi)
        x_cardi = self.pointwise_conv6(x_cardi)

        # Short Cut Branch
        x_out = self.group_conv7(x)
        x_out = self.pointwise_conv8(x_out)

        print("#------------------------------------------------------------")
        print("VarGNet Transition Block, Cardinality Channels: %d" % (self.first_increased_channels))
        print("VarGNet Transition Block, Increased_channels: %d" % (self.second_increased_channels))

        x_out = self.relu(x_cardi + x_out)

        return x_out

class VarGNet_explore_v2_UnitA(nn.Module):

    # """VarGNet v1 unit for stride=1"""
    # Variable Group Network, Stage Block, Highlight:
    # 1: Devire from mobilenet MobileNet
    # 2: Merge two depthwise seperable conv together
    # 3: Introduce classical residual block

    def __init__(self, in_channels, out_channels, image_size, strides=1, group_base=8, factor=2, is_shortcut=False, use_se=False, use_sk=False, use_se_global_conv=False, name=" ", **kwargs):
        super(VarGNet_explore_v2_UnitA, self).__init__()

        assert in_channels == out_channels
        assert in_channels % 8 == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = in_channels * factor
        self.group_base = group_base
        self.feature_map = image_size
        self.is_shortcut = is_shortcut
        self.relu = nn.ReLU(inplace=True)



            # First group conv takes place of depthwise conv
        self.group_conv1 = ConvBnRelu(in_channels=self.in_channels, out_channels=self.bottleneck_channels, kernel_size=3, strides=strides, padding=1, 
            groups=self.in_channels // self.group_base, is_bn_relu=True, name=name+"group_conv1")

            # First pointwise conv
        self.pointwise_conv2 = ConvBnRelu(in_channels=self.bottleneck_channels, out_channels=in_channels, 
            kernel_size=1, strides=1, padding=0, is_bn_relu=True, name=name+"pointwise_conv2")

            # Second group conv takes place of depthwise conv
        self.group_conv3 = ConvBnRelu(in_channels=in_channels, out_channels=self.bottleneck_channels, kernel_size=3, strides=1, padding=1, 
            groups=self.in_channels // self.group_base, is_bn_relu=True, name=name+"group_conv3")

            # Second pointwise conv
        self.pointwise_conv4 = ConvBnRelu(in_channels=self.bottleneck_channels, out_channels=out_channels, 
            kernel_size=1, strides=1, padding=0, is_bn_relu=True, name=name+"pointwise_conv4")

        if is_shortcut:
            self.group_conv5 = ConvBnRelu(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, strides=strides, padding=1, 
                groups=self.in_channels // self.group_base, is_bn_relu=True, name=name+"group_conv5")

                # First pointwise conv
            self.pointwise_conv6 = ConvBnRelu(in_channels=self.out_channels, out_channels=self.out_channels, 
                kernel_size=1, strides=1, padding=0, is_bn_relu=True, name=name+"pointwise_conv6")


    def hybrid_forward(self, x):
        print("###########################################################################################")
        print("VarGNet Stage Block, Input Feature Size: %d x %d" % (self.feature_map, self.feature_map))
        print("VarGNet Stage Block, Input Channels: %d" % (self.in_channels))
        print("VarGNet Stage Block, Group Base Channels: %d, Group Numbers: %d" % (self.group_base, self.in_channels // self.group_base))
        print("VarGNet Stage Block, Bottleneck Channels: %d" % (self.bottleneck_channels))

        print("#------------------------------------------------------------")
        out = self.group_conv1(x)
        out = self.pointwise_conv2(out)

        # ----------------------------------------------------------------------------------------------
        # script for train_imagenet_vargnet_explore_v2_1x_double_se_ratio2_cosine_batch128_epoch240_submit.sh
        # script for train_imagenet_vargnet_explore_v2_1x_double_se_ratio4_cosine_batch128_epoch240_submit.sh
        # if self.se:
        #    print("VarGNet Stage Block, This is for checking first SE_Block")
        #    out = self.se(out)
        # ----------------------------------------------------------------------------------------------

        out = self.group_conv3(out)
        out = self.pointwise_conv4(out)


        if self.is_shortcut:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("VarGNet ShortCut: takes place of input data with Conv(3x3) and Conv(1x1)")
            x = self.group_conv5(x)
            x = self.pointwise_conv6(x)

        out = self.relu(x + out)
        print("#------------------------------------------------------------")

        return out

class VarGNet_explore_V2(nn.Module):
    """
	VarGNet V2 model
	----------
	block : HybridBlock
		Class for the VarGNet block. Options are VarGNet V1, VarGNet V2.
	layers : list of int
		Numbers of layers in each block
	channels : list of int
		Numbers of channels in each block. Length should be one larger than layers list.
	classes : int, default 1000
		Number of classification classes.
	use_se : bool, default False
		Whether to use Squeeze-and-Excitation module
    """
    def __init__(self, channels, factor=2, group_base_channels=8, unitB_repeat_list=[1,1,1], unitA_repeat_list=[2,6,3], 
        classes=2, use_se=False, use_se_global_conv=False, **kwargs):
        super(VarGNet_explore_V2, self).__init__(**kwargs)

        assert len(unitB_repeat_list) == len(unitA_repeat_list)

        image_size = 256
        last_channels = int(max(channels/32, 1.0)*1024)


        
        self.features = nn.Sequential()

        self.features.add_module('convo', nn.Conv2D(in_channels=3, out_channels=channels, kernel_size=3, strides=2, padding=1, use_bias=False))
        self.features.add_module('bn0', nn.BatchNorm(channels))

            # self.features.add(nn.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding=1))

            # ----------------------------------------------------------------------------------------------
            # script for train_imagenet_vargnet_explore_v2_1x_headstage_se_cosine_batch128_epoch240_submit.sh
        self.features.add_module('UnitA_0', VarGNet_explore_v2_UnitA(in_channels=channels, out_channels=channels, image_size=image_size//2, strides=2, is_shortcut=True,
                         group_base=group_base_channels, factor=1, use_se_global_conv=False, use_se=use_se, name="VarGNet_explore_v2_head_block"))

            # script for train_imagenet_vargnet_explore_v2_1x_se_headstage_se_cosine_batch128_epoch240_submit.sh
            # self.features.add(VarGNet_explore_v2_UnitA(in_channels=channels, out_channels=channels, image_size=image_size//2, strides=2, is_shortcut=True,
            #            group_base=group_base_channels, factor=1, use_se_global_conv=use_se_global_conv, use_se=True, name="VarGNet_explore_v2_head_block"))
            # ----------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------
            # script for train_imagenet_vargnet_explore_v2_1x_stem_cosine_batch128_epoch240_submit.sh
            # self.features.add(Stem_Block(3, channels, image_size))
            # ----------------------------------------------------------------------------------------------

        image_size = image_size//4

        for i in range(len(unitB_repeat_list)):
            unitB_repeats, unitA_repeats = unitB_repeat_list[i], unitA_repeat_list[i]
            for j in range(unitB_repeats):

                    # self.features.add(VarGNet_explore_v2_UnitB(in_channels=channels, out_channels=channels*2, image_size=image_size, 
                    #     group_base=group_base_channels, factor=factor, use_se=False, name="VarGNet_explore_v2_Downsample_%d_%d_" %(i+1, j+1)))

                    # ----------------------------------------------------------------------------------------------
                    # script for train_imagenet_vargnet_explore_v2_1x_se_all_ratio2_cosine_batch128_epoch240_submit.sh
                self.features.add_module('stage%d_UnitB_%d'%(i+1, j+1), VarGNet_explore_v2_UnitB(in_channels=channels, out_channels=channels*2, image_size=image_size, strides=2,
                        group_base=group_base_channels, factor=factor, use_se=False, name="VarGNet_explore_v2_Downsample_%d_%d_" %(i+1, j+1)))
                    # ----------------------------------------------------------------------------------------------

                channels = channels*2
                image_size = image_size//2

            for k in range(unitA_repeats):
                self.features.add_module('stage%d_UnitA_%d'%(i+1, k+1), VarGNet_explore_v2_UnitA(in_channels=channels, out_channels=channels, image_size=image_size, strides=1,
                        group_base=group_base_channels, factor=factor, use_se_global_conv=use_se_global_conv, use_se=use_se, use_sk=False, name="VarGNet_explore_v2_Stage_%d_%d_"%(i+1, k+1)))

        self.features.add_module('head_conv0', nn.Conv2d(in_channels = channels, out_channels = last_channels, kernel_size=1, stride=1, padding=0))
        self.features.add('head_bn0', nn.BatchNorm2d(last_channels))
        self.features.add('head_relu0', nn.ReLU(inplace=True)

            # replace GlobalAvgPool2D with Global Depthwise Conv
            # self.features.add(nn.GlobalAvgPool2D())
        self.features.add('global_avg_pool', nn.Conv2d(in_channels = last_channels, out_channels = last_channels, kernel_size=image_size, stride=1, padding=0, groups=last_channels, use_bias=False))
        self.features.add('head_bn1', nn.BatchNorm(last_channels))
        self.features.add_module('head_relu1', nn.ReLU(inplace=True))

        self.output = nn.Linear(last_channels, classes)

    def hybrid_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x




vargnet_v1_specification ={ 0.25: (8, 2, [3,1], [2,1], int(32*0.25)),
                            0.5:  (8, 2, [3,1], [2,1], int(32*0.5)),
                            0.75: (8, 2, [3,1], [2,1], int(32*0.75)),
                              1 : (8, 2, [3,1], [2,1], int(32*1.0)),
                            1.25: (8, 2, [3,1], [2,1], int(32*1.25)),
                            1.5 : (8, 2, [3,1], [2,1], int(32*1.5)),
                            1.75: (8, 2, [3,1], [2,1], int(32*1.75)),
                              2 : (8, 2, [3,1], [2,1], int(32*2.0)),}

vargnet_v2_specification ={ 0.25: (8, 2, [1,1,1], [2,6,3], int(32*0.25)),
							0.5:  (8, 2, [1,1,1], [2,6,3], int(32*0.5)),
                            0.75: (8, 2, [1,1,1], [2,6,3], int(32*0.75)),
                              1 : (8, 2, [1,1,1], [2,6,3], int(32*1.0)),
                            1.25: (8, 2, [1,1,1], [2,6,3], int(32*1.25)),
                            1.5 : (8, 2, [1,1,1], [2,6,3], int(32*1.5)),
                            1.75: (8, 2, [1,1,1], [2,6,3], int(32*1.75)),
                              2 : (8, 2, [1,1,1], [2,6,3], int(32*2.0)),}


vargnet_version = [VarGNet_explore_V1, VarGNet_explore_V2]
vargnet_specification = [vargnet_v1_specification, vargnet_v2_specification]


def get_vargnet(version, pretrained=False, ctx = cpu(), number_level=1,
					root='~/.mxnet/models', **kwargs):
    '''
	Parameters
	----------
	version : int
		Version of VarGNet. Options are 1, 2.
	num_level : int
		level of model. Options are 0.25, 0.5, 0.75, 1.0, 2.0.
	pretrained : bool or str
		Boolean value controls whether to load the default pretrained weights for model.
		String value represents the hashtag for a certain version of pretrained weights.
	ctx : Context, default CPU
		The context in which to load the pretrained weights.
	root : str, default $MXNET_HOME/models
		Location for keeping the model parameters.
    '''
    assert 1 <= version <= 4, "Invalid vargnetnet version: %d. Options are 1, 2 and 3."%(version)
    vargnet_class = vargnet_version[version-1]
    vargnet_config = vargnet_specification[version-1]


    assert number_level in vargnet_config, "Invalid number of level: %d. Options are %s"%(number_level, str(vargnet_config.keys()))
    group_base_channels, expand_factor, unitB_repeat, unitA_repeat, inital_channels = vargnet_config[number_level]
    

    net = vargnet_class(channels=inital_channels, factor=expand_factor, group_base_channels=group_base_channels, 
        unitB_repeat_list=unitB_repeat, unitA_repeat_list=unitA_repeat, **kwargs)

    if pretrained:
        print('Not implement error')



    return net


def vargnet_explore_v2_0_25x(**kwargs):

    return get_vargnet(version=2, number_level=0.25, **kwargs)


def vargnet_explore_v2_0_5x(**kwargs):

    return get_vargnet(version=2, number_level=0.5, **kwargs)


def vargnet_explore_v2_0_75x(**kwargs):

    return get_vargnet(version=2, number_level=0.75, **kwargs)


def vargnet_explore_v2_1x(**kwargs):

    return get_vargnet(version=2, number_level=1.0, **kwargs)


def vargnet_explore_v2_1_25x(**kwargs):

    return get_vargnet(version=2, number_level=1.25, **kwargs)


def vargnet_explore_v2_1_5x(**kwargs):

    return get_vargnet(version=2, number_level=1.5, **kwargs)


def vargnet_explore_v2_1_75x(**kwargs):

    return get_vargnet(version=2, number_level=1.75, **kwargs)


def vargnet_explore_v2_2x(**kwargs):

    return get_vargnet(version=2, number_level=2, **kwargs)

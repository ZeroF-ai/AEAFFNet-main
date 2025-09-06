# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/MichaelFan01/STDC-Seg."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.registry import MODELS
from ..utils import resize
from .bisenetv1 import AttentionRefinementModule
from torch.nn import init

import math
import fvcore.nn.weight_init as weight_init
BatchNorm2d = nn.BatchNorm2d
class Atten(nn.Module):
    def __init__(self, in_chan, out_chan,ratio=16):
        super(FSChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        self.gn = nn.BatchNorm2d(in_chan, in_chan)

        self.fc1 = nn.Conv2d(in_chan, in_chan // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_chan // ratio, in_chan, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        weight_init.c2_xavier_fill(self.fc1)
        weight_init.c2_xavier_fill(self.fc2)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        avg_out = self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))))
        max_out = self.sigmoid(self.fc2(self.relu1(self.fc1(self.max_pool(x)))))
        feat_avg = torch.mul(x, avg_out)
        feat_max = torch.mul(x, max_out)
        x = feat_avg + feat_max
        feat = self.conv(x)
        return feat

class Fusion(BaseModule):
    """Feature Fusion Module to fuse low level output feature of Spatial Path
    and high level output feature of Context Path.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Feature Fusion Module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.Sigmoid())

    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        # Note: No BN and more 1x1 conv in paper.
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out
class DenseLayer4(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer4, self).__init__()
        self.bn = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1,bias=False)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class DenseBlock4(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock4, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        for i in range(num_layers):
            self.layers.append(DenseLayer4(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)

class Detail(nn.Module):
    def __init__(self, in_channels):
        super(Detail, self).__init__()

        self.dense = DenseBlock4(num_layers=3, in_channels=in_channels, growth_rate=32)
        self.conv3 = nn.Conv2d(352,256,3,padding=1)

    def forward(self, x):
        x =self.dense(x)
        x = self.conv3(x)
        return x
class MultiM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(MultiM, self).__init__()
        bn_mom = 0.1

        # 定义不同的扩张率
        dilations = [4,4,4,4] #例如4 4 4 4
        def _make_branch(d):
            return nn.Sequential(
                nn.Conv2d(inplanes, branch_planes, kernel_size=3, padding=d, dilation=d, groups=4, bias=False),
                BatchNorm(branch_planes, momentum=bn_mom),
                nn.ReLU(inplace=True)
            )

        self.branches = nn.ModuleList([_make_branch(d) for d in dilations])

        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, groups=branch_planes, bias=False),
                BatchNorm(branch_planes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes, branch_planes, kernel_size=1, bias=False),
            ) for _ in dilations
        ])
        self.base = nn.Sequential(
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(branch_planes * (len(dilations) + 1), outplanes, kernel_size=1, bias=False),
            BatchNorm(outplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
            BatchNorm(outplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        base_feat = self.base(x)
        branch_feats = []
        for i, branch in enumerate(self.branches):
            out = branch(x) + base_feat
            out = self.refine[i](out)
            branch_feats.append(out)
        feat = torch.cat([base_feat] + branch_feats, dim=1)
        out = self.fusion(feat) + self.shortcut(x)
        return out

class STDCModule(BaseModule):
    """STDCModule.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels before scaling.
        stride (int): The number of stride for the first conv layer.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layers.
        fusion_type (str): Type of fusion operation. Default: 'add'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        self.stride = stride
        self.with_downsample = True if self.stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        conv_0 = ConvModule(
            in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)

        if self.with_downsample:
            self.downsample = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)

            if self.fusion_type == 'add':
                self.layers.append(nn.Sequential(conv_0, self.downsample))
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2**(i + 1) if i != num_convs - 1 else 2**i
            self.layers.append(
                ConvModule(
                    out_channels // 2**i,
                    out_channels // out_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        if self.fusion_type == 'add':
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            inputs = self.skip(inputs)

        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)
        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_downsample:
                    x = layer(self.downsample(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)
@MODELS.register_module()
class AEAFFNet(BaseModule):
    """This backbone is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        stdc_type (int): The type of backbone structure,
            `STDCNet1` and`STDCNet2` denotes two main backbones in paper,
            whose FLOPs is 813M and 1446M, respectively.
        in_channels (int): The num of input_channels.
        channels (tuple[int]): The output channels for each stage.
        bottleneck_type (str): The type of STDC Module type, the value must
            be 'add' or 'cat'.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layer at each STDC Module.
            Default: 4.
        with_final_conv (bool): Whether add a conv layer at the Module output.
            Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> import torch
        >>> stdc_type = 'STDCNet1'
        >>> in_channels = 3
        >>> channels = (32, 64, 256, 512, 1024)
        >>> bottleneck_type = 'cat'
        >>> inputs = torch.rand(1, 3, 1024, 2048)
        >>> self = STDCNet(stdc_type, in_channels,
        ...                 channels, bottleneck_type).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 256, 128, 256])
        outputs[1].shape = torch.Size([1, 512, 64, 128])
        outputs[2].shape = torch.Size([1, 1024, 32, 64])
    """

    arch_settings = {
        'AEAFFNet2': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 num_convs=4,
                 with_final_conv=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert stdc_type in self.arch_settings, \
            f'invalid structure {stdc_type} for STDCNet.'
        assert bottleneck_type in ['add', 'cat'],\
            f'bottleneck_type must be `add` or `cat`, got {bottleneck_type}'

        assert len(channels) == 5,\
            f'invalid channels length {len(channels)} for STDCNet.'

        self.in_channels = in_channels
        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.prtrained = pretrained
        self.num_convs = num_convs
        self.with_final_conv = with_final_conv

        self.stages = ModuleList([
            ConvModule(
                self.in_channels,
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                self.channels[0],
                self.channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ])
        # `self.num_shallow_features` is the number of shallow modules in
        # `STDCNet`, which is noted as `Stage1` and `Stage2` in original paper.
        # They are both not used for following modules like Attention
        # Refinement Module and Feature Fusion Module.
        # Thus they would be cut from `outs`. Please refer to Figure 4
        # of original paper for more details.
        self.num_shallow_features = len(self.stages)

        for strides in self.stage_strides:
            idx = len(self.stages) - 1
            self.stages.append(
                self._make_stage(self.channels[idx], self.channels[idx + 1],
                                 strides, norm_cfg, act_cfg, bottleneck_type))
        # After appending, `self.stages` is a ModuleList including several
        # shallow modules and STDCModules.
        # (len(self.stages) ==
        # self.num_shallow_features + len(self.stage_strides))
        if self.with_final_conv:
            self.final_conv = ConvModule(
                self.channels[-1],
                max(1024, self.channels[-1]),
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def _make_stage(self, in_channels, out_channels, strides, norm_cfg,
                    act_cfg, bottleneck_type):
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                STDCModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride,
                    norm_cfg,
                    act_cfg,
                    num_convs=self.num_convs,
                    fusion_type=bottleneck_type))
        return Sequential(*layers)

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        outs = outs[self.num_shallow_features:]
        return tuple(outs)


@MODELS.register_module()
class AEAFFNetContextPathNet(BaseModule):
    """STDCNet with Context Path. The `outs` below is a list of three feature
    maps from deep to shallow, whose height and width is from small to big,
    respectively. The biggest feature map of `outs` is outputted for
    `STDCHead`, where Detail Loss would be calculated by Detail Ground-truth.
    The other two feature maps are used for Attention Refinement Module,
    respectively. Besides, the biggest feature map of `outs` and the last
    output of Attention Refinement Module are concatenated for Feature Fusion
    Module. Then, this fusion feature map `feat_fuse` would be outputted for
    `decode_head`. More details please refer to Figure 4 of original paper.

    Args:
        backbone_cfg (dict): Config dict for stdc backbone.
        last_in_channels (tuple(int)), The number of channels of last
            two feature maps from stdc backbone. Default: (1024, 512).
        out_channels (int): The channels of output feature maps.
            Default: 128.
        ffm_cfg (dict): Config dict for Feature Fusion Module. Default:
            `dict(in_channels=512, out_channels=256, scale_factor=4)`.
        upsample_mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``.
        align_corners (str): align_corners argument of F.interpolate. It
            must be `None` if upsample_mode is ``'nearest'``. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Return:
        outputs (tuple): The tuple of list of output feature map for
            auxiliary heads and decoder head.
    """

    def __init__(self,
                 backbone_cfg,
                 last_in_channels=(1024, 512),
                 out_channels=128,
                 ffm_cfg=dict(
                     in_channels=512, out_channels=256, scale_factor=4),
                 paraf_cfg=dict(
                      inplanes=1024,branch_planes=128,outplanes=128),                     
                 upsample_mode='nearest',
                 align_corners=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone_cfg)
        self.arms = ModuleList()
        self.convs = ModuleList()
        for channels in last_in_channels:
            self.arms.append(Atten(channels, out_channels))
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg))
        self.conv_avg = ConvModule(
            last_in_channels[0], out_channels, 1, norm_cfg=norm_cfg)

        self.ffm = Fusion(**ffm_cfg)
        self.sp =Detail(256)

        self.paraf = MultiM(**paraf_cfg)

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = list(self.backbone(x))
        feature_up = self.paraf(outs[-1])         
        arms_out = []
        for i in range(len(self.arms)):
            x_arm = self.arms[i](outs[len(outs) - 1 - i]) + feature_up
            feature_up = resize(
                x_arm,
                size=outs[len(outs) - 1 - i - 1].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
            feature_up = self.convs[i](feature_up)
            arms_out.append(feature_up)

        outs[0]=self.sp(outs[0])

        feat_fuse = self.ffm(outs[0], arms_out[1])

        # The `outputs` has four feature maps.
        # `outs[0]` is outputted for `STDCHead` auxiliary head.
        # Two feature maps of `arms_out` are outputted for auxiliary head.
        # `feat_fuse` is outputted for decoder head.
        outputs = [outs[0]] + list(arms_out) + [feat_fuse]
        return tuple(outputs)

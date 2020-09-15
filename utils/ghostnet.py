"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from utils.attention import SpatialAttention


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                  stride=1, padding=0, bias=True)
    ])
    return m


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1):
        super(ConvBnAct, self).__init__()
        self.convbnact = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        self.weights = 0.0

        x = self.convbnact(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
            #self.se = eca_layer(mid_chs)
            # self.se = None
        else:
            self.se = None
            #self.se = SpatialAttention()

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class Yolo_ghost(nn.Module):
    def __init__(self, config, width=1.0, dropout=0.2):
        super(Yolo_ghost, self).__init__()
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck

        # stage1
        layers = []
        output_channel = _make_divisible(16 * width, 4)
        hidden_channel = _make_divisible(16 * width, 4)
        layers.append(block(16, hidden_channel, output_channel, 3, 1, se_ratio=0))
        self.stage1 = nn.Sequential(*layers)

        # stage2
        layers = []
        output_channel = _make_divisible(24 * width, 4)
        hidden_channel = _make_divisible(48 * width, 4)
        layers.append(block(16, hidden_channel, output_channel, 3, 2, se_ratio=0))
        hidden_channel = _make_divisible(72 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        input_channel = output_channel
        self.stage2 = nn.Sequential(*layers)

        # stage3
        layers = []
        output_channel = _make_divisible(40 * width, 4)
        hidden_channel = _make_divisible(72 * width, 4)
        layers.append(block(input_channel, hidden_channel, output_channel, 3, 2, se_ratio=0.25))
        hidden_channel = _make_divisible(120 * width, 4)
        layers.append((block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0.25)))
        input_channel = output_channel
        self.stage3 = nn.Sequential(*layers)

        # stage4
        layers = []
        output_channel = _make_divisible(80 * width, 4)
        hidden_channel = _make_divisible(240 * width, 4)
        layers.append(block(input_channel, hidden_channel, output_channel, 3, 2, se_ratio=0))
        hidden_channel = _make_divisible(200 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        hidden_channel = _make_divisible(184 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        input_channel = output_channel
        output_channel = _make_divisible(112 * width, 4)
        hidden_channel = _make_divisible(480 * width, 4)
        layers.append(block(input_channel, hidden_channel, output_channel, 3, 1, se_ratio=0.25))
        hidden_channel = _make_divisible(672 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0.25))
        input_channel = output_channel
        self.stage4 = nn.Sequential(*layers)

        # stage5
        layers = []
        output_channel = _make_divisible(160 * width, 4)
        hidden_channel = _make_divisible(672 * width, 4)
        layers.append(block(input_channel, hidden_channel, output_channel, 5, 2, se_ratio=0.25))
        hidden_channel = _make_divisible(960 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0))
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0.25))
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0))
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0.25))
        layers.append(nn.Sequential(ConvBnAct(output_channel, 960, 1)))
        self.stage5 = nn.Sequential(*layers)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        # backbone
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x0 = self.stage3(x)
        x1 = self.stage4(x0)
        x2 = self.stage5(x1)
        return x


class stage1(nn.Module):
    def __init__(self, width=1.0):
        super(stage1, self).__init__()
        block = GhostBottleneck
        layers = []
        output_channel = _make_divisible(16 * width, 4)
        hidden_channel = _make_divisible(16 * width, 4)
        layers.append(block(16, hidden_channel, output_channel, 3, 1, se_ratio=0))
        self.stage1 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        return x


class stage2(nn.Module):
    def __init__(self, width=1.0):
        super(stage2, self).__init__()
        block = GhostBottleneck
        # stage2
        layers = []
        output_channel = _make_divisible(24 * width, 4)
        hidden_channel = _make_divisible(48 * width, 4)
        layers.append(block(16, hidden_channel, output_channel, 3, 2, se_ratio=0))
        hidden_channel = _make_divisible(72 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        self.stage2 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage2(x)
        return x


class stage3(nn.Module):
    def __init__(self, width=1.0):
        super(stage3, self).__init__()
        block = GhostBottleneck
        # stage2
        layers = []
        output_channel = _make_divisible(40 * width, 4)
        hidden_channel = _make_divisible(72 * width, 4)
        layers.append(block(24, hidden_channel, output_channel, 3, 2, se_ratio=0.25))
        hidden_channel = _make_divisible(120 * width, 4)
        layers.append((block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0.25)))
        self.stage3 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage3(x)
        return x


class stage4(nn.Module):
    def __init__(self, width=1.0):
        super(stage4, self).__init__()
        block = GhostBottleneck

        layers = []
        output_channel = _make_divisible(80 * width, 4)
        hidden_channel = _make_divisible(240 * width, 4)
        layers.append(block(40, hidden_channel, output_channel, 3, 2, se_ratio=0))
        hidden_channel = _make_divisible(200 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        hidden_channel = _make_divisible(184 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0))
        input_channel = output_channel
        output_channel = _make_divisible(112 * width, 4)
        hidden_channel = _make_divisible(480 * width, 4)
        layers.append(block(input_channel, hidden_channel, output_channel, 3, 1, se_ratio=0.25))
        hidden_channel = _make_divisible(672 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 3, 1, se_ratio=0.25))
        self.stage4 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage4(x)
        return x


class stage5(nn.Module):
    def __init__(self, width=1.0):
        super(stage5, self).__init__()
        block = GhostBottleneck

        layers = []
        output_channel = _make_divisible(160 * width, 4)
        hidden_channel = _make_divisible(672 * width, 4)
        layers.append(block(112, hidden_channel, output_channel, 5, 2, se_ratio=0.25))
        hidden_channel = _make_divisible(960 * width, 4)
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0))
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0.25))
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0))
        layers.append(block(output_channel, hidden_channel, output_channel, 5, 1, se_ratio=0.25))
        layers.append(nn.Sequential(
            nn.Conv2d(output_channel, 960, 1, 1, 1 // 2, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
        ))
        self.stage5 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage5(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


class ResNetConfig(NamedTuple):
    layers: 'list[int]'
    block: 'Union[BasicBlock,Bottleneck]'
    groups: 'int' = 1
    width_per_group: 'int' = 64
    pretrained_init: 'Optional' = None


ENCODER_ARCH = {
    'resnet18': ResNetConfig(
        [2, 2, 2, 2], BasicBlock, pretrained_init=models.resnet18),
    'resnet34': ResNetConfig(
        [3, 4, 6, 3], BasicBlock, pretrained_init=models.resnet34),
    'resnet50': ResNetConfig(
        [3, 4, 6, 3], Bottleneck, pretrained_init=models.resnet50),
    'resnet101': ResNetConfig(
        [3, 4, 23, 3], Bottleneck, pretrained_init=models.resnet101),
    'wide_resnet50_2': ResNetConfig(
        [3, 4, 6, 3], Bottleneck, width_per_group=64 * 2,
        pretrained_init=models.wide_resnet50_2),
    'wide_resnet101_2': ResNetConfig(
        [3, 4, 23, 3], Bottleneck, width_per_group=64 * 2,
        pretrained_init=models.wide_resnet101_2),
}


class ResNet(nn.Module):

    def __init__(self, block, layers, input_dim, output_dim, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_dim, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class FCNet(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3,
                 batch_norm=False):
        super().__init__()

        layers = [nn.Linear(
            input_dim,
            hidden_dims[0] if len(hidden_dims) > 0 else output_dim
        )]
        for i in range(len(hidden_dims)):
            layers.append(nn.ReLU(inplace=True))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.Linear(
                hidden_dims[i],
                hidden_dims[i + 1] if i + 1 < len(hidden_dims) else output_dim
            ))
            if i + 1 < len(hidden_dims):
                layers.append(nn.Dropout(dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FcResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))

    def forward(self, x):
        x2 = self.block(x)
        return x2 - x


class FCResNet(nn.Module):

    def __init__(self, in_dim, out_dim, num_blocks, hidden_dim, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(FcResidualBlock(hidden_dim, dropout))
        if out_dim is not None:
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FCResNetPoseDecoder(nn.Module):

    def __init__(self, emb_dim, num_blocks, hidden_dim, target_dims,
                 dropout=0):
        super().__init__()
        self.fc_res = FCResNet(
            emb_dim, None, num_blocks, hidden_dim, dropout=dropout)
        self.target_dict = {}
        for target_name, target_dim in target_dims:
            fc_out = nn.Linear(hidden_dim, target_dim)
            self.target_dict[target_name] = fc_out
            self.add_module('fc_{}'.format(target_name), fc_out)

    def forward(self, x, target_name):
        x = self.fc_res(x)
        return self.target_dict[target_name](x)


class FCPoseDecoder(nn.Module):

    def __init__(self, emb_dim, hidden_dims, target_dims, dropout=0):
        super().__init__()
        assert len(hidden_dims) >= 2
        last_fc_dim = hidden_dims[-1]
        self.fcn = FCNet(emb_dim, hidden_dims[:-1], last_fc_dim,
                         dropout=dropout, batch_norm=False)
        self.target_dict = {}
        for target_name, target_dim in target_dims:
            fc_out = nn.Linear(last_fc_dim, target_dim)
            self.target_dict[target_name] = fc_out
            self.add_module('fc_{}'.format(target_name), fc_out)

    def forward(self, x, target_name):
        x = F.relu(self.fcn(x))
        return self.target_dict[target_name](x)

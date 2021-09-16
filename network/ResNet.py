
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torch
from torch.nn import functional as F

import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Final_Model(nn.Module):

    def __init__(self, backbone_net, semantic_head):
        super(Final_Model, self).__init__()
        self.backend=backbone_net
        self.semantic_head=semantic_head

    def forward(self, x):
        middle_feature_maps=self.backend(x)
        
        semantic_output=self.semantic_head(middle_feature_maps)
      
        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN=if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(Bottleneck, self).__init__()
        self.if_BN=if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if self.if_BN:
            self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if self.if_BN:
            self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        if self.if_BN:
            self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.if_BN:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SemanticHead(nn.Module):

    def __init__(self,num_class=20,input_channel=1024):
        super(SemanticHead,self).__init__()
  
        self.conv_1=nn.Conv2d(input_channel, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu_1 = nn.LeakyReLU()

        self.conv_2=nn.Conv2d(512, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.LeakyReLU()

        self.semantic_output=nn.Conv2d(128, num_class, 1)

    def forward(self, input_tensor):
        res=self.conv_1(input_tensor)
        res=self.bn1(res)
        res=self.relu_1(res)
        
        res=self.conv_2(res)
        res=self.bn2(res)
        res=self.relu_2(res)
        
        res=self.semantic_output(res)
        return res

class ResNet_ASPP_1(nn.Module):

    def __init__(self, block, layers,if_BN,if_remission, if_range,zero_init_residual=False,norm_layer=None,groups=1, width_per_group=64):
        super(ResNet_ASPP_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN=if_BN
        self.if_remission=if_remission
        self.if_range=if_range

        self.inplanes = 128
        self.dilation = 1
   
        self.groups = groups
        self.base_width = width_per_group

        if self.if_remission and not self.if_range:
            self.conv1 = nn.Conv2d(4, 128, kernel_size=1, stride=1, padding=0,bias=True)
        if self.if_range and self.if_range:
            self.conv1 = nn.Conv2d(5, 128, kernel_size=1, stride=1, padding=0,bias=True)
        if not self.if_remission and not self.if_range:        
            self.conv1 = nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0,bias=True)


   
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU()
        

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.conv_Aspp_1=nn.Conv2d(768, 128, 3, padding=3, dilation=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu_1 = nn.LeakyReLU()
        self.conv_Aspp_2=nn.Conv2d(768, 128, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.LeakyReLU()
        
        self.conv_Aspp_3=nn.Conv2d(768, 128, 3, padding=9, dilation=9, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu_3 = nn.LeakyReLU()
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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
            if self.if_BN:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        outputs = {}
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
       
        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_1 = F.interpolate(x_1, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        '''
        outputs['stem'] = x
        outputs['res2'] = res_1
        outputs['res3'] = res_2
        outputs['res4'] = res_3
        outputs['res5'] = res_4
        '''
        res=[x,res_1,res_2,res_3,res_4]
        res=torch.cat(res, dim=1)
        res_1_1=self.conv_Aspp_1(res)
        res_1_1=self.bn1(res_1_1)
        res_1_1=self.relu_1(res_1_1)
        res_2_2=self.conv_Aspp_2(res)
        res_2_2=self.bn2(res_2_2)
        res_2_2=self.relu_2(res_2_2)
        res_3_3=self.conv_Aspp_3(res)
        res_3_3=self.bn3(res_3_3)
        res_3_3=self.relu_3(res_3_3)
        res_new=[res,res_1_1,res_2_2,res_3_3]
        return torch.cat(res_new, dim=1)

    def forward(self, x):
        return self._forward_impl(x)



class ResNet_ASPP_2(nn.Module):

    def __init__(self, block, layers,if_BN,if_remission, if_range,zero_init_residual=False,norm_layer=None,groups=1, width_per_group=64):
        super(ResNet_ASPP_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN=if_BN
        self.if_remission=if_remission
        self.if_range=if_range

        self.inplanes = 128
        self.dilation = 1
   
        self.groups = groups
        self.base_width = width_per_group

        if self.if_remission and not self.if_range:
            self.conv1 = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0,bias=True)
        if self.if_range and self.if_range:
            self.conv1 = nn.Conv2d(5, 64, kernel_size=1, stride=1, padding=0,bias=True)
        if not self.if_remission and not self.if_range:        
            self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,bias=True)


   
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU()
        

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        self.conv_Aspp_1=nn.Conv2d(128*7, 256, 3, padding=3, dilation=3, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.LeakyReLU()
        self.conv_Aspp_2=nn.Conv2d(128*7, 256, 3, padding=6, dilation=6, bias=True)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu_2 = nn.LeakyReLU()
        
        self.conv_Aspp_3=nn.Conv2d(128*7, 256, 3, padding=9, dilation=9, bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu_3 = nn.LeakyReLU()
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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
            if self.if_BN:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        outputs = {}
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
       
        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_1 = F.interpolate(x_1, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        '''
        outputs['stem'] = x
        outputs['res2'] = res_1
        outputs['res3'] = res_2
        outputs['res4'] = res_3
        outputs['res5'] = res_4
        '''
        res=[x,res_1,res_2,res_3,res_4]
        res=torch.cat(res, dim=1)
        res_1_1=self.conv_Aspp_1(res)
        res_1_1=self.bn1(res_1_1)
        res_1_1=self.relu_1(res_1_1)
        res_2_2=self.conv_Aspp_2(res)
        res_2_2=self.bn2(res_2_2)
        res_2_2=self.relu_2(res_2_2)
        res_3_3=self.conv_Aspp_3(res)
        res_3_3=self.bn3(res_3_3)
        res_3_3=self.relu_3(res_3_3)
        res_new=[res,res_1_1,res_2_2,res_3_3]
        return torch.cat(res_new, dim=1)

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_34_point(nn.Module):

    def __init__(self, block, layers,if_BN,if_remission, if_range, with_normal,zero_init_residual=False,norm_layer=None,groups=1, width_per_group=64):
        super(ResNet_34_point, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN=if_BN
        self.if_remission=if_remission
        self.if_range=if_range
        self.with_normal=with_normal

        self.inplanes = 512
        self.dilation = 1
   
        self.groups = groups
        self.base_width = width_per_group

        if not self.if_remission and not self.if_range and not self.with_normal:        
            self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and not self.if_range and not self.with_normal:
            self.conv1 = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0,bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and self.if_range and not self.with_normal:
            self.conv1 = nn.Conv2d(5, 64, kernel_size=1, stride=1, padding=0,bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and self.if_range and self.with_normal:
            self.conv1 = nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0,bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()



   
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn_2 = nn.BatchNorm2d(512)
        self.relu_2 = nn.LeakyReLU()
        

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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
            if self.if_BN:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        outputs = {}
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn_0(x)
        x = self.relu_0(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv4(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
       
        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        #res_1 = F.interpolate(x_1, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        res=[x,x_1,res_2,res_3,res_4]

        return torch.cat(res, dim=1)

    def forward(self, x):
        return self._forward_impl(x)







def _resnet(arch, block, layers, if_BN,if_remission):
    model = ResNet(block, layers, if_BN,if_remission,if_range,zero_init_residual=False)

    return model

def _resnet_aspp(arch, block, layers, if_BN,if_remission,if_range):
    model = ResNet_ASPP(block, layers, if_BN,if_remission,if_range,zero_init_residual=False)

    return model



def _resnet_aspp_1(arch, block, layers, if_BN,if_remission,if_range):
    model = ResNet_ASPP_1(block, layers, if_BN,if_remission,if_range,zero_init_residual=False)

    return model

def _resnet_aspp_2(arch, block, layers, if_BN,if_remission,if_range):
    model = ResNet_ASPP_2(block, layers, if_BN,if_remission,if_range,zero_init_residual=False)

    return model

def _resnet_point(arch, block, layers, if_BN,if_remission,if_range,with_normal):
    model = ResNet_34_point(block, layers, if_BN,if_remission,if_range,with_normal,zero_init_residual=False)

    return model

def resnet18(if_BN,if_remission,if_range):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], if_BN,if_remission,if_range)



def resnet18_aspp(if_BN,if_remission,if_range):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_aspp('resnet18', BasicBlock, [2, 2, 2, 2], if_BN,if_remission,if_range)



def resnet34_aspp_1(if_BN,if_remission,if_range):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_aspp_1('resnet34',BasicBlock, [3, 4, 6, 3], if_BN,if_remission,if_range)

def resnet34_aspp_2(if_BN,if_remission,if_range):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_aspp_2('resnet34',BasicBlock, [3, 4, 6, 3], if_BN,if_remission,if_range)

def resnet34_point(if_BN,if_remission,if_range,with_normal):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_point('resnet34',BasicBlock, [3, 4, 6, 3], if_BN,if_remission,if_range,with_normal)




def resnet34(if_BN,if_remission):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], if_BN,if_remission)


def resnet50(pretrained=False, progress=True):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

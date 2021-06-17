import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, layer_name, block_index, \
        layer_input, layer_kernel, layer_stride, layer_padding, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        layer_kernel['layer%s.%s.conv1' %(layer_name, block_index)] = 3
        layer_stride['layer%s.%s.conv1' %(layer_name, block_index)] = stride
        layer_padding['layer%s.%s.conv1' %(layer_name, block_index)] = 1

        layer_kernel['layer%s.%s.conv2' %(layer_name, block_index)] = 3
        layer_stride['layer%s.%s.conv2' %(layer_name, block_index)] = stride
        layer_padding['layer%s.%s.conv2' %(layer_name, block_index)] = 1

        
        self.layer_input = layer_input
        self.layer_name = layer_name
        self.block_index = block_index
        # self.exist_downsample = False

    def forward(self, x):
        residual = x
        self.layer_input['layer%s.%s.conv1' %(self.layer_name, self.block_index)] = x.data
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        self.layer_input['layer%s.%s.conv2' %(self.layer_name, self.block_index)] = out.data
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            self.layer_input['layer%s.%s.downsample.0' %(self.layer_name, self.block_index)] = x.data
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer_name, block_index, \
        layer_input, layer_kernel, layer_stride, layer_padding,  stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        layer_kernel['layer%s.%s.conv1' %(layer_name, block_index)] = 1
        layer_stride['layer%s.%s.conv1' %(layer_name, block_index)] = 1
        layer_padding['layer%s.%s.conv1' %(layer_name, block_index)] = 1

        layer_kernel['layer%s.%s.conv2' %(layer_name, block_index)] = 3
        layer_stride['layer%s.%s.conv2' %(layer_name, block_index)] = stride
        layer_padding['layer%s.%s.conv2' %(layer_name, block_index)] = 1

        layer_kernel['layer%s.%s.conv3' %(layer_name, block_index)] = 1
        layer_stride['layer%s.%s.conv3' %(layer_name, block_index)] = 1
        layer_padding['layer%s.%s.conv3' %(layer_name, block_index)] = 1

        self.layer_input = layer_input
        self.layer_name = layer_name
        self.block_index = block_index

    def forward(self, x):
        residual = x
        self.layer_input['layer%s.%s.conv1' %(self.layer_name, self.block_index)] = x.data
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        self.layer_input['layer%s.%s.conv2' %(self.layer_name, self.block_index)] = out.data
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        self.layer_input['layer%s.%s.conv3' %(self.layer_name, self.block_index)] = out.data
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            self.layer_input['layer%s.%s.downsample.0' %(self.layer_name, self.block_index)] = x.data
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # Modified by Chen Shangyu to get layer inputs
        self.layer_input = dict()        
        self.layer_kernel = {'conv1': 7}
        self.layer_stride = {'conv1': 2}
        self.layer_padding = {'conv1': 3}

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], layer_name='1')
        self.layer2 = self._make_layer(block, 128, layers[1], layer_name='2', stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], layer_name='3', stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], layer_name='4', stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, layer_name, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            self.layer_kernel['layer%s.0.downsample.0' %layer_name] = 1
            self.layer_stride['layer%s.0.downsample.0' %layer_name] = stride
            self.layer_padding['layer%s.0.downsample.0' %layer_name] = 0

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # def __init__(self, inplanes, planes, layer_name, block_index, \
        #   layer_input, layer_kernel, layer_stride, layer_padding, stride=1, downsample=None):
        layers = []
        layers.append(block(self.inplanes, planes, layer_name, block_index = 0,
            layer_input = self.layer_input,
            layer_kernel = self.layer_kernel, 
            layer_stride = self.layer_stride, 
            layer_padding = self.layer_padding, 
            stride = stride, 
            downsample = downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_name, block_index = i,
                        layer_input = self.layer_input,
                        layer_kernel = self.layer_kernel, 
                        layer_stride = self.layer_stride, 
                        layer_padding = self.layer_padding))

        return nn.Sequential(*layers)

    def forward(self, x):

        self.layer_input['conv1'] = x.data
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self.layer_input['fc'] = x.data
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

# model.py 예시

import torch
import torch.nn as nn
from torchvision.models import resnet50

class CSRNet_ResnetDilated(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet_ResnetDilated, self).__init__()

        # 1) ResNet50을 불러옴
        backbone = resnet50(pretrained=True)

        # 2) layer1, layer2, layer3, layer4 중, layer4의 stride를 1로 바꾸고 dilation을 2로 설정
        #    이렇게 하면 최종 downsampling 배수가 x8이 됨 (기본 x32 → x8).
        #    layer4 내부 블록들의 downsample 모듈과 convolution stride들을 수정해야 함.
        
        # stem + layer1 + layer2 + layer3 까지는 그대로
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # layer4 stride/dilation 수정
        self.layer4 = backbone.layer4
        # 원본 layer4 블록은 stride=2로 downsampling
        # -> 이를 stride=1, dilation=2로 바꿔주어 최종 해상도 x8 유지
        for n, module in self.layer4.named_modules():
            if 'conv2' in n:
                # conv2에서 dilation=2
                module.dilation = (2,2)
                module.padding = (2,2)
            elif 'downsample.0' in n:
                # downsample conv의 stride=1
                module.stride = (1,1)
            elif 'conv2' not in n and 'downsample.0' in n:
                # downsample에서 stride를 1로
                module.stride = (1,1)
            elif 'conv1' in n or 'conv3' in n:
                # conv1, conv3는 stride=1 그대로
                pass
        
        # 3) layer4까지 통과하면 출력 채널이 2048, 해상도는 입력의 1/8
        #    그 다음에 CSRNet backend + output layer를 달아주면 됨
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=2048, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.layer0(x)   # /4
        x = self.layer1(x)   # /4
        x = self.layer2(x)   # /8
        x = self.layer3(x)   # /16
        x = self.layer4(x)   # /16 → dilation=2 적용하여 receptive field 확대, 최종 해상도는 /8
        x = self.backend(x)  # CSRNet의 backend
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

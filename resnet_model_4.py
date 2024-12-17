import torch
import torch.nn as nn
from torchvision import models

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

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # ----- Dilated ResNet50: 최종 출력 해상도 /8 만들기 -----
        backbone = models.resnet50(pretrained=True)

        # layer3: stride=2 -> stride=1, dilation=2
        for name, module in backbone.layer3.named_modules():
            if 'conv2' in name:
                module.dilation = (2, 2)
                module.padding = (2, 2)
            if 'downsample.0' in name or 'conv2' in name:
                if hasattr(module, 'stride'):
                    module.stride = (1, 1)

        # layer4: stride=2 -> stride=1, dilation=4
        for name, module in backbone.layer4.named_modules():
            if 'conv2' in name:
                module.dilation = (4, 4)
                module.padding = (4, 4)
            if 'downsample.0' in name or 'conv2' in name:
                if hasattr(module, 'stride'):
                    module.stride = (1, 1)

        # stem + layer1 + layer2 + layer3 + layer4 = /8 해상도
        self.frontend = nn.Sequential(
            backbone.conv1,   # /2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, # /4
            backbone.layer1,  # /4
            backbone.layer2,  # /8
            backbone.layer3,  # /8 (dilation=2, stride=1)
            backbone.layer4   # /8 (dilation=4, stride=1)
        )
        
        # 백엔드는 기존 CSRNet과 동일, 입력 채널 2048
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=2048, dilation=True)
        
        # 최종 출력: 1채널 density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # 초기화
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)      # (B, 2048, H/8, W/8)
        x = self.backend(x)       # (B, 64,    H/8, W/8)
        x = self.output_layer(x)  # (B, 1,     H/8, W/8)
        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

import torch
import torch.nn as nn
from torchvision import models

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    원본 CSRNet 코드와 동일한 함수 시그니처/동작을 유지.
    backend 생성에 활용한다.
    """
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
        
        # ========== Dilated ResNet50 백본 세팅 ==========
        backbone = models.resnet50(pretrained=True)
        
        # layer4에 dilation=2, stride=1로 수정해서 최종 출력 해상도가 x8이 되도록 만들기
        # (기존 ResNet50은 x32가 기본)
        for name, module in backbone.layer4.named_modules():
            # conv2에서 dilation=2 적용
            if 'conv2' in name:
                module.dilation = (2,2)
                module.padding = (2,2)
            # downsample 또는 layer4 첫 블록의 stride를 1로 변경
            if 'downsample.0' in name or 'conv2' in name:
                if hasattr(module, 'stride'):
                    module.stride = (1,1)
        
        # stem부터 layer4까지를 frontend로 묶는다
        # (원본 CSRNet frontend처럼 "x = self.frontend(x)" 한 번에 통과)
        self.frontend = nn.Sequential(
            backbone.conv1,   # /2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, # /4
            backbone.layer1,  # /4
            backbone.layer2,  # /8
            backbone.layer3,  # /16
            backbone.layer4   # /16 -> dilation=2로 receptive field만 늘리고, 해상도는 /8 유지
        )
        
        # ========== Backend는 기존 CSRNet 구조 그대로 ==========
        # 단, ResNet50 최종 출력 채널은 2048이므로 backend의 입력 채널도 2048로 설정
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=2048, dilation=True)
        
        # 출력층 (1채널 밀도 맵)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # ========== Weight Init ==========
        # vgg16처럼 특정 레이어 가중치를 복사하는 대신,
        # ResNet50의 pretrained weights는 이미 로드됨.
        # backend와 output_layer는 별도로 초기화.
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)      # Dilated ResNet50
        x = self.backend(x)       # CSRNet backend
        x = self.output_layer(x)  # 최종 1채널 density map
        return x

    def _initialize_weights(self):
        """
        원본 CSRNet 코드와 동일하게 Conv2d 레이어를 normal_(std=0.01)로 초기화.
        여기서는 backend와 output_layer가 대상.
        """
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

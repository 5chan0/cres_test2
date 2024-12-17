import torch
import torch.nn as nn
from torchvision import models

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    기존 CSRNet의 backend를 생성하는 함수.
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
        
        # ========== ConvNeXt-small 백본 세팅 ==========
        convnext = models.convnext_small(pretrained=True)
        
        # ConvNeXt-small의 특징 추출부 (출력 채널: 768, 출력 해상도: /32)
        self.frontend = convnext.features  # (B, 768, H/32, W/32)
        
        # ========== Backend는 기존 CSRNet과 동일하게 ==========
        # 단, ConvNeXt-small의 출력 채널 수인 768로 변경
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=768, dilation=True)
        
        # 최종 출력층: 1채널 밀도 맵
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # ========== Weight Initialization ==========
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)      # (B, 768, H/32, W/32)
        x = self.backend(x)       # (B, 64, H/32, W/32)
        x = self.output_layer(x)  # (B, 1, H/32, W/32)
        return x

    def _initialize_weights(self):
        """
        기존 CSRNet과 동일하게 backend와 output_layer의 가중치를 초기화.
        """
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

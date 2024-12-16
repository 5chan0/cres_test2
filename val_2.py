import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter  # filters 제거
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch

from torchvision import datasets, transforms

########################
# 수정된 부분: transforms 정의
########################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

root = './'

# 경로 설정 (ShanghaiTech 등)
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
# 실제로 사용할 폴더
path_sets = [part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

model = CSRNet()

# CUDA 가용 여부 확인
if torch.cuda.is_available():
    model = model.cuda()

# 미리 학습된 모델 가중치 로드 (체크포인트)
checkpoint = torch.load('0model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

mae = 0.0

for i in range(len(img_paths)):
    # ---------------------------
    # 수정된 부분: Image.open → transforms로 처리
    # ---------------------------
    pil_img = Image.open(img_paths[i]).convert('RGB')
    img = transform(pil_img)  # 훈련 시와 동일한 전처리 파이프라인
    if torch.cuda.is_available():
        img = img.cuda()

    # Ground truth .h5 파일 로드
    gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
    groundtruth = np.asarray(gt_file['density'])
    
    # 모델 추론
    output = model(img.unsqueeze(0))  # (1, C, H, W)

    # MAE 계산
    # output.sum() => 예측된 총 카운트, groundtruth.sum() => 실제 사람 수
    mae += abs(output.detach().cpu().sum().numpy() - np.sum(groundtruth))
    
    print(i, mae)  # 진행도 확인용

final_mae = mae / len(img_paths)
print("Final MAE:", final_mae)

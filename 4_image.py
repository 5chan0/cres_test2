import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
import cv2

def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    
    # ----- 1) 이미지/GT를 32의 배수로 패딩 -----
    w, h = img.size
    pad_w = 0 if w % 32 == 0 else (32 - (w % 32))
    pad_h = 0 if h % 32 == 0 else (32 - (h % 32))
    if pad_w > 0 or pad_h > 0:
        new_w = w + pad_w
        new_h = h + pad_h
        padded_img = Image.new('RGB', (new_w, new_h), (0,0,0))
        padded_img.paste(img, (0,0))
        img = padded_img

    # ----- 2) GT density map 로드 및 동일 패딩 -----
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    gt_file.close()

    H, W = target.shape
    if pad_w > 0 or pad_h > 0:
        new_target = np.zeros((H + pad_h, W + pad_w), dtype=target.dtype)
        new_target[:H, :W] = target
        target = new_target

    # ----- 3) /32로 리사이즈 후, *1024 스케일 -----
    new_w = target.shape[1] // 32
    new_h = target.shape[0] // 32
    target = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    target = target * 1024  # 32 * 32

    return img, target

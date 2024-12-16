import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    gt_file.close()

    # 아래 블록은 if False로 묶여있어 실행되지 않으므로 주석 처리 또는 필요 시 수정 가능
    """
    if False:
        crop_size = (img.size[0] // 2, img.size[1] // 2)
        if random.randint(0, 9) <= -1:
            dx = int(random.randint(0, 1)*img.size[0]/2.0)
            dy = int(random.randint(0, 1)*img.size[1]/2.0)
        else:
            dx = int(random.random()*img.size[0]/2.0)
            dy = int(random.random()*img.size[1]/2.0)

        img = img.crop((dx, dy, crop_size[0]+dx, crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy, dx:crop_size[0]+dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    """

    # Python 3에서는 cv2.resize의 인자에 정수형 튜플을 넣어야 하므로 // 연산으로 수정
    new_w = target.shape[1] // 8
    new_h = target.shape[0] // 8
    target = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_CUBIC) * 64

    return img, target

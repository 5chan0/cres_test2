import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch

# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    # zip으로 좌표생성. Python 3에서는 list 필요하면 list(zip(...)) 사용
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[int(pt[1]), int(pt[0])] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 4.0  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

root = '.'

# now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

# 아래는 A 파트 ground truth 생성
path_sets = [part_A_train, part_A_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images','ground_truth').replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images','ground_truth'), 'w') as hf:
        hf['density'] = k

# 시각화 예시
plt.imshow(Image.open(img_paths[0]))

gt_file = h5py.File(img_paths[0].replace('.jpg', '.h5').replace('images','ground_truth'), 'r')
groundtruth = np.asarray(gt_file['density'])
gt_file.close()
plt.imshow(groundtruth, cmap=CM.jet)

print(np.sum(groundtruth))

# 이제 ShanghaiB의 ground truth 생성
path_sets = [part_B_train, part_B_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images','ground_truth').replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    # 원본 코드대로 B 파트는 sigma=15로 고정
    k = gaussian_filter(k, 15)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images','ground_truth'), 'w') as hf:
        hf['density'] = k

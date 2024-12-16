import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import load_data
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            # 학습 데이터 증강을 위해 리스트를 4배로 늘리는 로직(원본 코드를 그대로 유지)
            root = root * 4
        
        # list만 shuffle 가능하므로 root가 list 타입이어야 함
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index < len(self), 'index range error' 
        img_path = self.lines[index]
        
        img, target = load_data(img_path, self.train)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

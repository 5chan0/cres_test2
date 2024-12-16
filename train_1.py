import sys
import os

import warnings

from model import CSRNet
from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

# 파이썬 3에서 print함수 사용
parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')

def main():
    global args, best_prec1
    best_prec1 = 1e6
    
    args = parser.parse_args()
    
    # LR을 1e-5로 시작
    args.original_lr = 1e-5
    args.lr = 1e-5
    
    args.batch_size = 1
    args.momentum = 0.95       # AdamW에서는 beta 파라미터 사용되지만, 여기서는 그대로 둡니다.
    args.decay = 5e-4
    args.start_epoch = 0
    
    # 총 epoch를 200으로 설정
    args.epochs = 200
    
    # steps, scales는 외부 스케줄링 사용 시 크게 의미 없음
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    
    # json 파일 읽기
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(int(args.seed))
    
    model = CSRNet()
    model = model.cuda()
    
    criterion = nn.MSELoss(reduction='sum').cuda()
    
    # AdamW 최적화 기법 사용
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    # 변경: CosineAnnealingLR 스케줄러 사용
    # 초기 lr=1e-5 -> eta_min=1e-7 로 200 epoch에 걸쳐 부드럽게 감소
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    # mae_log.txt 초기화 (신규 작성)
    with open('mae_log.txt', 'w') as f:
        f.write('epoch,MAE\n')
    
    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, model, criterion, optimizer, epoch, args)
        
        # epoch 끝난 뒤 검증
        prec1 = validate(val_list, model, criterion, args)
        
        # mae_log.txt에 기록
        with open('mae_log.txt', 'a') as f:
            f.write('{}, {:.5f}\n'.format(epoch, prec1))
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)
        
        # CosineAnnealing 스케줄러 업데이트
        scheduler.step()

def train(train_list, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True, 
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    
    # 현재 lr 표시를 위해 optimizer.param_groups[0]에서 lr 추적
    current_lr = optimizer.param_groups[0]['lr']
    print('epoch {}, processed {} samples, lr {:.10f}'.format(
        epoch, epoch * len(train_loader.dataset), current_lr))
    
    model.train()
    end = time.time()
    
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        
        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_list, model, criterion, args):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=False),
        batch_size=args.batch_size)
    
    model.eval()
    mae = 0.0
    
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.cuda()
            img = Variable(img)
            output = model(img)
            mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
    
    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))
    return mae

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()

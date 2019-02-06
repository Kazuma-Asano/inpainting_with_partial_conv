#coding:utf-8
from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim

from   torch.utils.data       import DataLoader
from   torchvision.utils      import save_image, make_grid
from   networks               import PConvUNet, VGG16FeatureExtractor
from   loss                   import InpaintingLoss
from   dataloader             import get_training_set#, get_val_set
# from   util                   import progress_bar
import torch.backends.cudnn as cudnn

import pandas as pd
import matplotlib.pyplot as plt

#####################################################################################
LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch Inpainting with Partial Convolution')
    parser.add_argument('--batchSize', '-b', type=int, default=16, help='training batch size')
    parser.add_argument('--testBatchSize', '-tb', type=int, default=4, help='testing batch size')
    parser.add_argument('--nEpochs', '-e', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--mask_nc', type=int, default=3, help='mask image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=2e-4')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=100, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    option = parser.parse_args()
    print(option)
    return option

def train(model, train_data_loader, criterion, optimizer):
    model.train()
    for iteration, batch in enumerate(train_data_loader):
        input, mask, gt = batch[0], batch[1], batch[2]
        # ここ要確認
        input.requires_grad_()
        mask.requires_grad_()

        if opt.cuda:
            input = input.cuda()
            mask = mask.cuda()
            gt = gt.cuda()
        output, _ = model(input, mask)

        loss_dict = criterion(input, mask, output, gt)

        loss = 0.0
        for key, coef in LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO:progress_bar追加


if __name__ == '__main__':

    opt = get_parser()
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("GPU is not found, please run without --cuda")

    cudnn.benchmark = True

    gpu_ids = []

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        gpu_ids = [0]

    #################################################################################

    root_path = './dataset/'
    train_set = get_training_set(root_path)
    # val_set = get_val_set(root_path)

    train_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                    batch_size=opt.batchSize, shuffle=True)

    # val_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads,
    #                                 batch_size=opt.testBatchSize, shuffle=False)

    model = PConvUNet()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    criterion = InpaintingLoss(VGG16FeatureExtractor())

    if opt.cuda:
        model.cuda()
        criterion.cuda()

    for epoch in range(1, opt.nEpochs + 1):
        print('\n Epoch:{}'.format(epoch))
        train(model, train_data_loader, criterion, optimizer)

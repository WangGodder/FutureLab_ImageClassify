from __future__ import print_function, absolute_import
import argparse
import os
import shutil
import time
from utils.logging import Logger
from loss.CenterLoss import CenterLoss
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime


from visdom import Visdom
import numpy as np
global plotter

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='challenge'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

plotter = VisdomLinePlotter(env_name='challenge')

import sys
import csv
sys.path.append('.')
import XTmodelzoo
from model import utils
from datasets import MyTestDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext101_64x4d')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--data', metavar='DIR', default=os.path.join('..','DataProcess','testB','data'),
                    help='path to dataset')
parser.add_argument('--model-path', default=os.path.join('.','temp','resnext101_64x4d2018_05_16_15_41_31.pth.tar_model_best.pth.tar'), type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--result-path', default=os.path.join('.','result','resnext101_64x4d2018_05_16_15_41_31.pth.tar_model_best.csv'), type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
best_prec1 = 0


def main():
    global args, best_prec1, t ,loss_weight, Identity_num
    t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    loss_weight = 0.001
    Identity_num = 'Null'

    args = parser.parse_args()
    sys.stdout = Logger(os.path.join('.', 'logs', args.arch + t + '.txt'))
    # create model
    print("=> creating model '{}'".format(args.arch))
    print("=> --learning-rate '{}'".format(args.lr))
    print("=> --momentum '{}'".format(args.momentum))
    print("=> --weight_decay '{}'".format(args.weight_decay))
    print("=> --batch-size '{}'".format(args.batch_size))
    print("=> --pretrained '{}'".format(args.pretrained))
    print("=> --loss_weight '{}'".format(loss_weight))
    print("=> --Identity_num '{}'".format(Identity_num))
    print("=> --model_path '{}'".format(args.model_path))
    print("=> --result_path '{}'".format(args.result_path))


    model = XTmodelzoo.__dict__[args.arch](num_classes=20, pretrained=args.pretrained)

    # optionally resume from a checkpoint

    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)

        best_prec1 = checkpoint['best_prec1']
        new_dict = {k[7:] : v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(new_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model_path, checkpoint['epoch']))
    else:
        print("=> no model found at '{}'".format(args.model_path))

    cudnn.benchmark = True

    # Data loading code
    testdir = os.path.join(args.data)

    scale = 0.875

    print('Images transformed from size {} to {}'.format(
        int(round(max(model.input_size) / scale)),
        model.input_size))

    test_tf = utils.TransformImage(model, scale=scale)

    test_loader = torch.utils.data.DataLoader(
        MyTestDataset(data_root=testdir,csv_path=os.path.join('..','DataProcess','testB','list.csv'), transform=test_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    nllloss = nn.NLLLoss().cuda()
    # Center loss
    loss_weight = 0.001
    center_loss = CenterLoss(20, 20).cuda()

    criterion = [nllloss, center_loss]

    optimizer4nn = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimzer4center = torch.optim.SGD(center_loss.parameters(), lr=0.5)
    optimizer = [optimizer4nn, optimzer4center]

    model = torch.nn.DataParallel(model).cuda()

    test(test_loader, model, criterion, args.result_path)

def transform(cate):
    if cate == '0':
        return '12'
    if cate == '1':
        return '4'
    if cate == '2':
        return '1'
    if cate == '3':
        return '19'
    if cate == '4':
        return '0'
    if cate == '5':
        return '3'
    if cate == '6':
        return '2'
    if cate == '7':
        return '17'
    if cate == '8':
        return '14'
    if cate == '9':
        return '5'
    if cate == '10':
        return '15'
    if cate == '11':
        return '13'
    if cate == '12':
        return '9'
    if cate == '13':
        return '18'
    if cate == '14':
        return '11'
    if cate == '15':
        return '10'
    if cate == '16':
        return '8'
    if cate == '17':
        return '7'
    if cate == '18':
        return '16'
    if cate == '19':
        return '6'


def write_pred_csv(output, topk, path, writer):
    maxk = max(topk)
    batch_size = output.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    top = pred.cpu().numpy().tolist()
    #rows = [[]]
    for p in range(0,len(top)):
        writer.writerow([path[p].split('\\')[-1].split('.')[0], transform(str(top[p][0])), transform(str(top[p][1])),
                         transform(str(top[p][2]))])

def test(test_loader, model, criterion, save_path):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    csvfile = open(save_path, "w",newline='')
    csvfile.writer = csv.writer(csvfile)
    # 先写入columns_name
    #csvfile.writer.writerow(['FILE_ID','CATEGORY_ID_TOP3'])
        # 写入多行用writerows
    #csvfile.writer.writerows([[0, 1, 3], [1, 2, 3], [2, 3, 4]])
    # switch to evaluate mode
    model.eval()

    end = time.time()
    csvfile.writer.writerow(['FILE_ID','CATEGORY_ID0','CATEGORY_ID1','CATEGORY_ID2'])   # csv title
    for i, (input, path) in enumerate(test_loader):
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        ip, pred = model(input_var)
        write_pred_csv(output=pred, topk=(1, 3), path = path, writer = csvfile.writer)
    csvfile.close()


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
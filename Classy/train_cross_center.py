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

sys.path.append('.')
import XTmodelzoo
from model import utils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default=os.path.join('..','DataProcess','data'),
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext101_64x4d')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',
                    default=False,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')

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
    model = XTmodelzoo.__dict__[args.arch](num_classes=20, pretrained=args.pretrained)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            new_dict = {k[7:] : v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(new_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
         datasets.ImageFolder(traindir, transforms.Compose([
             transforms.RandomSizedCrop(max(model.input_size)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize,
        ])),
         batch_size=args.batch_size, shuffle=True,
         num_workers=args.workers, pin_memory=True)

    scale = 0.875

    print('Images transformed from size {} to {}'.format(
        int(round(max(model.input_size) / scale)),
        model.input_size))

    val_tf = utils.TransformImage(model, scale=scale)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    # Nll loss
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

    if args.evaluate:
        prec1 = validate(val_loader, model, criterion)
        print(prec1)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        top1,top3,lossnn,losscenter =train(train_loader, model, criterion, optimizer, epoch, loss_weight)

        # print("train_acc1: ", top1)
        # plotter.plot('acc', 'train_acc1', epoch, top1)
        # plotter.plot('acc', 'train_acc3', epoch, top3)
        # plotter.plot('loss_nn', 'train_loss_nn', epoch, lossnn)
        # plotter.plot('loss_center', 'train_loss_center', epoch, losscenter)



        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        plotter.plot('loss_nn', 'test_loss_nn', epoch, float(prec1[0].data))
        plotter.plot('loss_center', 'test_loss_center', epoch, float(prec1[1].data))

        # remember best prec@1 and save checkpoint
        is_best = float(prec1[0].data) > best_prec1
        best_prec1 = max(float(prec1[0].data), best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best,  os.path.join('.', 'temp', args.arch + t + '.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, loss_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_nn = AverageMeter()
    losses_ceter = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    ip1_loader = []
    idx_loader = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        ip1, pred = model(input_var)
        #ip1, pred = model(data)
        loss1 = criterion[0](pred, target)
        loss2 = criterion[1](target, ip1)
        # print(loss1)
        # print(loss2)

        loss = loss1 + loss_weight * loss2
        # loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        prec1, prec3 = accuracy(pred.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        losses_nn.update(loss1.data[0],input.size(0))
        losses_ceter.update(loss2.data[0],input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        loss.backward()
        # optimizer.step()
        optimizer[0].step()
        optimizer[1].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                  'loss@1 {losses_nn.val:.4f} ({losses_nn.avg:.4f})\t'
                  'loss@2 {losses_ceter.val:.4f} ({losses_ceter.avg:.4f}))'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3,
                losses_nn=losses_nn, losses_ceter=losses_ceter))

    return top1, top3, losses_nn, losses_ceter


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        ip, pred = model(input_var)

        loss = criterion[0](pred, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(pred.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top3=top3))

    print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return top1.avg, top3.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + '_model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer[0].param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
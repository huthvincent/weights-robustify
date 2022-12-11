import argparse
import logging
import pdb
import sys
import time
import math
from tqdm import tqdm
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from utils import *

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu)/std


upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='./data/cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--fname', default='cifar_pretrain', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    return parser.parse_args()

def config_lr_scheduler(args):
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]
    return lr_schedule

def main():
    args = get_args()

    # redirect output to ./output directory
    args.fname = os.path.join('./output', args.fname, str(args.seed))
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))

    dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        proxy = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(28, 10, widen_factor=args.width_factor, dropRate=0.0)
        proxy = WideResNet(28, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    proxy = nn.DataParallel(proxy).cuda()
    # model.to(device)
    # proxy.to(device)

    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    lr_schedule = config_lr_scheduler(args)

    best_test_robust_acc = 0
    if args.resume:
        state_resumed = torch.load(os.path.join(args.fname, f'state.pth'))
        model.load_state_dict(state_resumed['model_state'])
        opt.load_state_dict(state_resumed['opt_state'])
        start_epoch = state_resumed['epoch']
        logger.info(f'Resuming at epoch {start_epoch}')
        best_test_robust_acc = state_resumed['test_acc']
    else:
        start_epoch = 0

    epochs = args.epochs + start_epoch

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    if not args.eval:
        logger.info(f'{"="*20} Train {"="*20}')
        logger.info('Epoch \t Time Elapse \t LR \t \t Loss \t Acc')
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            model.train()
            for i, batch in enumerate(tqdm(train_batches)):
                X, y = batch['input'], batch['target']
                lr = lr_schedule(epoch + (i + 1) / len(train_batches))
                opt.param_groups[0].update(lr=lr)

                output = model(normalize(X))
                loss = criterion(output, y)

                # L1 regularization
                if args.l1:
                    for name,param in model.named_parameters():
                        if 'bn' not in name and 'bias' not in name:
                            loss += args.l1*param.abs().sum()

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)

            train_time = time.time()
            logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr,
                train_loss/train_n, train_acc/train_n)

            # save checkpoint upon validation
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:      
                model.eval()
                test_loss = 0
                test_acc = 0
                test_n = 0
                for i, batch in enumerate(tqdm(test_batches)):
                    X, y = batch['input'], batch['target']

                    output = model(normalize(X))
                    loss = criterion(output, y)

                    test_loss += loss.item() * y.size(0)
                    test_acc += (output.max(1)[1] == y).sum().item()
                    test_n += y.size(0)

                logger.info(f'{"="*20} Test {"="*20}')
                logger.info('Loss \t Acc')
                logger.info('%.4f \t %.4f', test_loss/test_n, test_acc/test_n)

                if test_acc/test_n > best_test_robust_acc:
                    logger.info('Saving model')
                    torch.save({
                            'model_state':model.state_dict(),
                            'opt_state':opt.state_dict(),
                            'epoch':epoch,
                            'test_acc':test_acc/test_n,
                        }, os.path.join(args.fname, f'state.pth'))
                    best_test_robust_acc = test_acc/test_n

        logger.info('Finish Training')


if __name__ == "__main__":
    main()

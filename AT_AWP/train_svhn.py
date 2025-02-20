import argparse
import logging
import pdb
import sys
import time
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from utils_awp import AdvWeightPerturb

svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.5, 0.5, 0.5)

mu = torch.tensor(svhn_mean).view(3,1,1).cuda()
std = torch.tensor(svhn_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.uniform_(-0.5,0.5).renorm(p=2, dim=1, maxnorm=epsilon)
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./svhn-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise'])
    parser.add_argument('--lr-max', default=0.01, type=float)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=1, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='svhn_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=5, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

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

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    num_workers = 2
    train_dataset = datasets.SVHN(
        args.data_dir, split='train', transform=train_transform, download=True)
    test_dataset = datasets.SVHN(
        args.data_dir, split='test', transform=test_transform, download=True)

    val_ratio = 0.2
    length = len(train_dataset)
    train_length = int((1 - val_ratio) * length)
    # train_indices, val_indices = [subset.indices for subset in random_split(train_dataset, [train_length, length - train_length])]
    train_dataset, val_set = random_split(train_dataset, [train_length, length - train_length])

    # pdb.set_trace()

    # split train and validation
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # model = models_dict[args.architecture]().cuda()
    # model.apply(initialize_weights)
    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        proxy = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
        proxy = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = model.cuda()
    proxy = proxy.cuda()

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
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
        # lr_schedule = lambda t: np.interp([t], [0, args.epochs], [0, args.lr_max])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.

    best_val_robust_loss = math.inf
    if args.resume:
        start_epoch = args.resume
        state_resumed = torch.load(os.path.join(args.fname, f'state_{start_epoch-1}.pth'))
        model.load_state_dict(state_resumed['model_state'])
        opt.load_state_dict(state_resumed['opt_state'])
        logger.info(f'Resuming at epoch {start_epoch}')

        best_val_robust_loss = state_resumed['val_robust_loss']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    if not args.eval:
        logger.info(f'{"="*20} Train {"="*20}')
        logger.info('Epoch \t Time Elapse \t LR \t \t Loss \t Acc \t Robust Loss \t Robust Acc')
        for epoch in range(start_epoch, epochs):
            model.train()
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_robust_loss = 0
            train_robust_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(tqdm(train_loader)):
                X, y = X.cuda(), y.cuda()
                if args.mixup:
                    X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                    X, y_a, y_b = map(Variable, (X, y_a, y_b))
                lr = lr_schedule(epoch + (i + 1) / len(train_loader))
                opt.param_groups[0].update(lr=lr)

                if args.attack == 'pgd':
                    # Random initialization
                    if args.mixup:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                    delta = delta.detach()

                # Standard training
                elif args.attack == 'none':
                    delta = torch.zeros_like(X)
                X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

                model.train()
                # calculate adversarial weight perturbation and perturb it
                if epoch >= args.awp_warmup:
                    # not compatible to mixup currently.
                    assert (not args.mixup)
                    awp = awp_adversary.calc_awp(inputs_adv=X_adv,
                                                targets=y)
                    awp_adversary.perturb(awp)

                robust_output = model(X_adv)
                if args.mixup:
                    robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
                else:
                    robust_loss = criterion(robust_output, y)

                if args.l1:
                    for name,param in model.named_parameters():
                        if 'bn' not in name and 'bias' not in name:
                            robust_loss += args.l1*param.abs().sum()

                opt.zero_grad()
                robust_loss.backward()
                opt.step()

                if epoch >= args.awp_warmup:
                    awp_adversary.restore(awp)

                output = model(normalize(X))
                if args.mixup:
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                else:
                    loss = criterion(output, y)

                train_robust_loss += robust_loss.item() * y.size(0)
                train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)

            train_time = time.time()

            logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n)

            # save checkpoint upon validation
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                model.eval()

                val_loss = 0
                val_acc = 0
                val_robust_loss = 0
                val_robust_acc = 0
                val_n = 0

                for i, (X, y) in enumerate(tqdm(val_loader)):
                    X, y = X.cuda(), y.cuda()

                    # Random initialization
                    if args.attack == 'none':
                        delta = torch.zeros_like(X)
                    else:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
                    delta = delta.detach()

                    robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    robust_loss = criterion(robust_output, y)

                    output = model(normalize(X))
                    loss = criterion(output, y)

                    val_robust_loss += robust_loss.item() * y.size(0)
                    val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                    val_loss += loss.item() * y.size(0)
                    val_acc += (output.max(1)[1] == y).sum().item()
                    val_n += y.size(0)

                logger.info(f'{"="*20} Validation {"="*20}')
                logger.info('Loss \t Acc \t Robust Loss \t Robust Acc')
                logger.info('%.4f \t %.4f \t %.4f \t %.4f', val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                if val_robust_loss/val_n < best_val_robust_loss:
                    torch.save({
                            'model_state':model.state_dict(),
                            'opt_state':opt.state_dict(),
                            'val_robust_acc':val_robust_acc/val_n,
                            'val_robust_loss':val_robust_loss/val_n
                        }, os.path.join(args.fname, f'state_{epoch}.pth'))
                    best_val_robust_loss = val_robust_loss/val_n
        logger.info('Finish Training')


    model.eval()
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    for i, (X, y) in enumerate(tqdm(test_loader)):
        X, y = X.cuda(), y.cuda()

        # Random initialization
        if args.attack == 'none':
            delta = torch.zeros_like(X)
        else:
            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
        delta = delta.detach()

        robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss = criterion(robust_output, y)

        output = model(normalize(X))
        loss = criterion(output, y)

        test_robust_loss += robust_loss.item() * y.size(0)
        test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)

    logger.info(f'{"="*20} Test {"="*20}')
    logger.info('Loss \t Acc \t Robust Loss \t Robust Acc')
    logger.info('%.4f \t %.4f \t %.4f \t %.4f', test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)



if __name__ == "__main__":
    main()

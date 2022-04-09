from __future__ import print_function
import os
import sys
import argparse
import random
from turtle import Vec2D
import visdom
import json
import time
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from model import CAUNet, SqueezeWB
from dataset import AverageMeter, get_angular_loss, evaluate, WBDataset
from tqdm import tqdm, trange


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='CAUNet',
                        choices=['CAUNet', 'SqueezeWB'])
    parser.add_argument('--cameras',
                        type=str,
                        nargs='+')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=16,
                        help='input batch size')
    parser.add_argument('--nepoch',
                        type=int,
                        default=2000,
                        help='number of epochs to train for')
    parser.add_argument('--workers',
                        type=int,
                        help='number of data loading workers',
                        default=24)
    parser.add_argument('--lrate',
                        type=float,
                        default=5e-5,
                        # default=3e-4,
                        help='learning rate')
    parser.add_argument('--env',
                        type=str,
                        default='main',
                        help='visdom environment')
    parser.add_argument('--pth_path', type=str, default='')
    parser.add_argument('--foldnum', type=int, default=0, help='fold number')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    opt = get_args()

    # init save dir
    now = datetime.datetime.now()
    save_path = now.isoformat()
    dir_name = './log/{}'.format(opt.env)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logname = os.path.join(dir_name, 'log_fold' + str(opt.foldnum) + '.txt')

    # visualization
    vis = visdom.Visdom(port=8097, env=opt.env + '-' + save_path)
    win_curve = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
    )

    train_loss = AverageMeter()
    val_loss = AverageMeter()

    # load data
    print('training fold %d' % opt.foldnum)
    dataset_train = WBDataset(camera_list=opt.cameras, train=True, fold=opt.foldnum)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.workers)
    len_dataset_train = len(dataset_train)
    print('len_dataset_train:', len(dataset_train))
    dataset_test = WBDataset(camera_list=opt.cameras, train=False, fold=opt.foldnum)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)
    len_dataset_test = len(dataset_test)
    print('len_dataset_test:', len(dataset_test))

    # create network
    if opt.model == 'CAUNet':
        model = CAUNet().cuda()
    else:
        model = SqueezeWB().cuda()
    if opt.pth_path != '':
        print('loading pretrained model')
        model.load_state_dict(torch.load(opt.pth_path))

    # optimizer
    lrate = opt.lrate
    optimizer = optim.Adam(model.parameters(), lr=lrate)

    # train
    print('start train.....')
    best_val_loss = 100.0
    for epoch in trange(opt.nepoch):
        # train mode
        train_loss.reset()
        model.train()
        for i, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            img, label, fn = data
            img, label = img.cuda(), label.cuda()
            pred = model(img)
            loss = get_angular_loss(pred, label)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())

        vis.line(X=np.array([epoch]),
                 Y=np.array([train_loss.avg]),
                 win=win_curve,
                 name='train loss',
                 update='append')

        # val mode
        if epoch % 20 == 0:
            with torch.no_grad():
                val_loss.reset()
                model.eval()
                errors = []
                for i, data in enumerate(dataloader_test):
                    img, label, fn = data
                    img, label = img.cuda(), label.cuda()
                    pred = model(img)
                    loss = get_angular_loss(pred, label)
                    val_loss.update(loss.item())
                    errors.append(loss.item())
                vis.line(X=np.array([epoch]),
                         Y=np.array([val_loss.avg]),
                         win=win_curve,
                         name='val loss',
                         update='append')

            mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
            print('Epoch: %d,  Train_loss: %f,  Val_loss: %f' %
                  (epoch, train_loss.avg, val_loss.avg))
            if (val_loss.avg > 0 and val_loss.avg < best_val_loss):
                best_val_loss = val_loss.avg
                torch.save(model.state_dict(),
                           '%s/fold%d.pth' % (dir_name, opt.foldnum))

            log_table = {
                "train_loss": train_loss.avg,
                "val_loss": val_loss.avg,
                "epoch": epoch,
                "lr": lrate,
                "best_val_loss": val_loss.avg,
                "mean": mean,
                "median": median,
                "trimean": trimean,
                "bst25": bst25,
                "wst25": wst25,
                "pct95": pct95
            }
            with open(logname, 'a') as f:
                f.write('json_stats: ' + json.dumps(log_table) + '\n')

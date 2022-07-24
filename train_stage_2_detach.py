from __future__ import print_function
import os
from re import A
import sys
import argparse
import random
from turtle import Vec2D
# import visdom
import json
import time
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from model import SqueezeWB, ReSqueezeWB
from dataset import AverageMeter, get_angular_loss, evaluate, WBDataset, ColorChecker
from tqdm import tqdm, trange

ITER_NUM = 3
GAMMA = 0.8


def get_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-p', '--pth_path', type=str, default='')
    parser.add_argument('--foldnum', type=int, default=0, help='fold number')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    opt = get_args()

    # init save dir
    now = datetime.datetime.now()
    save_path = now.isoformat()
    dir_name = './log_newagu_detach/{}'.format(opt.env)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logname = os.path.join(dir_name, 'log_fold' + str(opt.foldnum) + '.txt')

    # # visualization
    # vis = visdom.Visdom(port=8097, env=opt.env + '-' + save_path)
    # win_curve = vis.line(
    #     X=np.array([0]),
    #     Y=np.array([0]),
    # )

    train_loss = AverageMeter()
    val_loss = AverageMeter()

    # load data
    # print('training fold %d' % opt.foldnum)
    # dataset_train = WBDataset(camera_list=opt.cameras,
    #                           train=True, fold=opt.foldnum)
    # dataloader_train = torch.utils.data.DataLoader(dataset_train,
    #                                                batch_size=opt.batch_size,
    #                                                shuffle=True,
    #                                                num_workers=opt.workers)
    # len_dataset_train = len(dataset_train)
    # print('len_dataset_train:', len(dataset_train))
    # dataset_test = WBDataset(camera_list=opt.cameras,
    #                          train=False, fold=opt.foldnum)
    # dataloader_test = torch.utils.data.DataLoader(dataset_test,
    #                                               batch_size=1 if 'canon1d' in opt.cameras else opt.batch_size,
    #                                               shuffle=False,
    #                                               num_workers=1 if 'canon1d' in opt.cameras else opt.workers,
    #                                               drop_last=False)
    # len_dataset_test = len(dataset_test)
    # print('len_dataset_test:', len(dataset_test))

#load data
    dataset_train = ColorChecker(train=True,folds_num=opt.foldnum)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,shuffle=True, num_workers=opt.workers)
    len_dataset_train = len(dataset_train)
    print('len_dataset_train:',len(dataset_train))

    dataset_test = ColorChecker(train=False,folds_num=opt.foldnum)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,shuffle=True, num_workers=opt.workers)
    len_dataset_test = len(dataset_test)
    print('len_dataset_test:',len(dataset_test))
    print('training fold %d' % opt.foldnum)

    # create network
    base_model = SqueezeWB().cuda()
    model = ReSqueezeWB().cuda()
    if opt.pth_path != '':
        print('loading pretrained model from{}:' .format(opt.pth_path))
        base_model.load_state_dict(torch.load(opt.pth_path))
        model.load_state_dict(torch.load(opt.pth_path), strict=False) 

    # optimizer
    lrate = opt.lrate
    # optimizer = optim.Adam([dict(params=base_model.parameters()), 
    #                         dict(params=model.parameters())], 
    #                         lr=lrate)
    optimizer = optim.AdamW([dict(params=base_model.parameters()), 
                            dict(params=model.parameters())], 
                            lr=lrate)
    exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.5)
    #调整学习率

    # train
    print('start train.....')
    best_val_loss = 100.0
    for epoch in trange(opt.nepoch):
        # train mode
        train_loss.reset()
        base_model.eval()
        model.train()
        for i, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            img, label, fn = data
            img, label = img.cuda(), label.cuda()

            # initial predict
            with torch.no_grad():
                pred = base_model(img) #(32x3)
            pred = pred/pred[:, 1][:, None] #(32x3/32x1),逐元素除法
            pred_list = [pred] #打包在一个长为1的list中
            loss_list = []
           

            # recurrent
            lstm_memory = (None, None, None)
            for iter in range(ITER_NUM):
                _img = img/pred_list[-1][..., None, None] #img:[16,3,512,512] pred_list:torch.Size([16, 3, 1, 1])
                _img = torch.clamp(_img, 0, 1)
                pred, lstm_memory = model(_img, lstm_memory)
                pred = pred/pred[:, 1][:, None]
                pred = pred * pred_list[-1]
                pred_list.append(pred)

            loss_list += [GAMMA**(ITER_NUM-1-iter)*get_angular_loss(pred, label) for iter, pred in enumerate(pred_list[1:])]
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())

        exp_lr_scheduler.step()

        # vis.line(X=np.array([epoch]),
        #          Y=np.array([train_loss.avg]),
        #          win=win_curve,
        #          name='train loss',
        #          update='append')

        # val mode
        if epoch % 20 == 0:
            with torch.no_grad():
                val_loss.reset()
                base_model.eval()
                model.eval()
                errors = []
                for i, data in enumerate(dataloader_test):
                    img, label, fn = data
                    img, label = img.cuda(), label.cuda()

                    # initial predict
                    pred = base_model(img)
                    pred = pred/pred[:, 1][:, None]
                    pred_list = [pred]

                    # recurrent
                    lstm_memory = (None, None, None)
                    for iter in range(ITER_NUM):
                        _img = img/pred_list[-1][..., None, None]
                        _img = torch.clamp(_img, 0, 1)
                        pred, lstm_memory = model(_img, lstm_memory)
                        pred = pred/pred[:, 1][:, None]
                        pred = pred * pred_list[-1]
                        pred_list.append(pred)

                    # print([torch.nn.functional.normalize(pred[0][None], dim=1) for pred in pred_list], label[0][None])
                    pred = pred_list[-1]

                    loss = get_angular_loss(pred, label)
                    val_loss.update(loss.item(), n=img.shape[0])
                    errors += [get_angular_loss(p[None], l[None]).item() for p, l in zip(pred, label)]

                # vis.line(X=np.array([epoch]),
                #          Y=np.array([val_loss.avg]),
                #          win=win_curve,
                #          name='val loss',
                #          update='append')

            mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
            print('Epoch: %d,  Train_loss: %f,  Val_loss: %f' %
                  (epoch, train_loss.avg, val_loss.avg))
            if (val_loss.avg > 0 and val_loss.avg < best_val_loss):
                best_val_loss = val_loss.avg
                torch.save(base_model.state_dict(),
                           '%s/base_fold%d.pth' % (dir_name, opt.foldnum))
                torch.save(model.state_dict(),
                           '%s/fold%d.pth' % (dir_name, opt.foldnum))

            log_table = {
                "train_loss": train_loss.avg,
                "val_loss": val_loss.avg,
                "epoch": epoch,
                "lr": lrate,
                "best_val_loss": best_val_loss,
                "mean": mean,
                "median": median,
                "trimean": trimean,
                "bst25": bst25,
                "wst25": wst25,
                "pct95": pct95
            }
            with open(logname, 'a') as f:
                f.write('json_stats: ' + json.dumps(log_table) + '\n')
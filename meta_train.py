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
from model import SqueezeWB, ReSqueezeWB
from dataset import AverageMeter, get_angular_loss, evaluate, MetaWBDataset
from tqdm import tqdm, trange
import learn2learn as l2l


ITER_NUM = 3
GAMMA = 0.8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',
                        type=int,
                        help='k shots')
    parser.add_argument('--tps',
                        type=int,
                        default=32,
                        help='tasks per step')
    parser.add_argument('--fas',
                        type=int,
                        default=5,
                        help='fast adaption steps')
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
    parser.add_argument('--mamlrate',
                        type=float,
                        default=1e-2,
                        help='meta learning rate')
    parser.add_argument('--env',
                        type=str,
                        default='main',
                        help='visdom environment')
    parser.add_argument('-bp', '--base_pth_path', type=str, default='.pretrain/nus_stage_2_base.pth')
    parser.add_argument('-p', '--pth_path', type=str, default='.pretrain/nus_stage_2.pth')
    parser.add_argument('--foldnum', type=int, default=0, help='fold number')
    opt = parser.parse_args()
    print(opt)
    return opt


def meta_step(opt, base_model, meta_model, dataloader_query, dataloader_val):
    learner = meta_model.clone()
    train_data = []
    for i, data in enumerate(dataloader_query):
        if i == opt.k:
            break
    train_data.append(data)

    # train k shot with fas steps
    for _ in range(opt.k):
        for data in train_data:
            img, label, fn = data
            img, label = img.cuda(), label.cuda()
                
            pred_list = []
            # initial predict
            with torch.no_grad():
                pred = base_model(img)
                pred = pred/pred[:, 1][:, None] #？确认一下维度
            pred_list = [pred]

            # recurrent
            lstm_memory = (None, None, None)
            for _ in range(ITER_NUM):
                _img = img/pred_list[-1][..., None, None]
                _img = torch.clamp(_img, 0, 1) #将所有元素限制在01区间
                pred, lstm_memory = learner(_img, lstm_memory)
                pred = pred/pred[:, 1][:, None]
                pred = pred * pred_list[-1]
                pred_list.append(pred)
            loss = sum([GAMMA**(ITER_NUM-1-iter)*get_angular_loss(pred, label) for iter, pred in enumerate(pred_list[1:])])
            learner.adapt(loss)

        # compute validation loss
        validate_loss = 0.
        validate_counter = 0
        validate_error_list = []
        for i, data in enumerate(dataloader_val):
            img, label, fn = data
            img, label = img.cuda(), label.cuda()
                
            pred_list = []
            # initial predict
            with torch.no_grad():
                pred = base_model(img)
                pred = pred/pred[:, 1][:, None]
            pred_list = [pred]

            # recurrent
            lstm_memory = (None, None, None)
            for _ in range(ITER_NUM):
                _img = img/pred_list[-1][..., None, None]
                _img = torch.clamp(_img, 0, 1)
                pred, lstm_memory = learner(_img, lstm_memory)
                pred = pred/pred[:, 1][:, None]
                pred = pred * pred_list[-1]
                pred_list.append(pred)
            loss = sum([GAMMA**(ITER_NUM-1-iter)*get_angular_loss(pred, label) for iter, pred in enumerate(pred_list[1:])])
            validate_loss += loss
            validate_counter += 1
            validate_error_list.append(get_angular_loss(pred_list[-1], label).item())
        validate_loss /= validate_counter
    return validate_loss, validate_error_list


if __name__ == '__main__':
    opt = get_args()

    # init save dir
    now = datetime.datetime.now() #获取系统时间
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
    ) #远点位置x=0,y=0

    train_loss = AverageMeter()
    val_loss = AverageMeter()

    # load data
    print('training fold %d' % opt.foldnum)
    # load train-query data
    dataloader_train_list = []
    for camera in ['canon_eos_1D_mark3', 'canon_eos_600D', 'fuji', 'nikonD5200', 'olympus', 'panasonic', 'samsung', 'sony']:
        dataset_train = MetaWBDataset(camera_list=[camera],
                                      train=True, fold=opt.foldnum)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       num_workers=1)
        dataloader_train_list.append(dataloader_train)

    # load train-val data
    dataloader_train_val_list = []
    for camera in ['canon_eos_1D_mark3', 'canon_eos_600D', 'fuji', 'nikonD5200', 'olympus', 'panasonic', 'samsung', 'sony']:
        dataset_train_val = MetaWBDataset(camera_list=[camera],
                                      train=False, fold=opt.foldnum)
        dataloader_train_val = torch.utils.data.DataLoader(dataset_train_val,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=1) 
        dataloader_train_val_list.append(dataloader_train_val) #raw, illumination, meta['img_name']    

    # load test-query data
    dataloader_test_list = []
    for camera_list in [['nikonD40'], ['canon1d', 'canon5d']]:
        dataset_test = MetaWBDataset(camera_list=camera_list,
                                      train=True, fold=opt.foldnum)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       num_workers=1)
        dataloader_test_list.append(dataloader_test)

    # load test-val data
    dataloader_test_val_list = []
    for camera_list in [['nikonD40'], ['canon1d', 'canon5d']]:
        dataset_test_val = MetaWBDataset(camera_list=camera_list,
                                      train=True, fold=opt.foldnum)
        dataloader_test_val = torch.utils.data.DataLoader(dataset_test_val,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=1)
        dataloader_test_val_list.append(dataloader_test_val)


    # create network
    base_model = SqueezeWB().cuda()
    model = ReSqueezeWB().cuda()
    if opt.pth_path != '':
        print('loading pretrained model from: {}'.format(opt.pth_path))
        base_model.load_state_dict(torch.load(opt.base_pth_path))
        model.load_state_dict(torch.load(opt.pth_path), strict=False)
    base_model.eval()
    meta_model = l2l.algorithms.MAML(model, lr=opt.mamlrate) #？什么鬼东西

    # optimizer
    lrate = opt.lrate
    optimizer = optim.AdamW(meta_model.parameters(), lr=lrate)

    # train
    print('start train.....')
    best_val_loss = 100.0
    for epoch in trange(opt.nepoch):
        # train mode
        train_loss.reset()
        meta_model.train()
        
        tps_error = 0.
        for _ in range(opt.tps):
            task_idx = np.random.randint(0, len(dataloader_train_list))
            validate_loss, _ = meta_step(opt, base_model, meta_model, dataloader_train_list[task_idx], dataloader_train_val_list[task_idx])
            tps_error += validate_loss
        tps_error /= opt.tps            
        optimizer.zero_grad()
        tps_error.backward()
        optimizer.step() #计算梯度，反向传播

        train_loss.update(tps_error.item()) #更新train_loss

        vis.line(X=np.array([epoch]),
                 Y=np.array([train_loss.avg]),
                 win=win_curve,
                 name='train loss',
                 update='append')

        # val mode
        if epoch % 20 == 0:
            with torch.no_grad(): #val与test之前都要先有torch.no_grad!!
                val_loss.reset()
                meta_model.eval()
                _, errors = meta_step(opt, base_model, meta_model, dataloader_test_list[0], dataloader_test_val_list[0])
                val_loss.avg = sum(errors)/len(errors)

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

#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
foldnum=1
env=squeezenet_ccd_stage2_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
# python train_stage_2.py -b 16 --nepoch 2000 --foldnum $foldnum --env $env --cameras canon1d canon5d -p 'pretrain/log_new_agu/1stage_ourccm/fold'$foldnum'.pth'
python train_stage_2.py -b 16 --nepoch 2000 --foldnum $foldnum --env $env  -p '/ys/YS_prjs/MyWork/C4-master/results/0802_addmask/fold'$foldnum'.pth'
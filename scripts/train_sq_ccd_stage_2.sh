#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
foldnum=2
env=squeezenet_ccd_stage2_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
python train_stage_2.py -b 16 --nepoch 2000 --foldnum $foldnum --env $env --cameras canon1d canon5d -p 'pretrain/C4_stage1_ourccm/log/C4_sq_1stage_ourccm/fold2.pth'
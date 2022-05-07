#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
foldnum=1
env=squeezenet_ccd_stage2_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
python train_stage_2.py -b 16 --nepoch 2000 --foldnum $foldnum --env $env --cameras canon1d canon5d -p ../pretrain/squeezenet_ccd_stage1_fold1.2022-05-03_07-17-28/fold1.pth

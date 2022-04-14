#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
foldnum=0
env=squeezenet_ccd_stage2_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
python train_stage_2.py -b 16 --nepoch 2000 --foldnum $foldnum --env $env --cameras canon1d canon5d -p .pretrain/ccd_stage_1.pth

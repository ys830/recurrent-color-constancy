#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
foldnum=0
env=caunet_stage1_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
python train_stage_1.py --nepoch 2000 --foldnum $foldnum --env $env --model CAUNet --cameras canon1d canon5d

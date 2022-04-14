#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
foldnum=1
env=squeezenet_ccd_stage1_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
python train_stage_1.py -b 32 --nepoch 2000 --foldnum $foldnum --env $env --model SqueezeWB --cameras canon1d canon5d

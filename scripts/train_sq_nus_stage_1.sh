#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
foldnum=0
env=squeezenet_nus_stage1_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
python train_stage_1.py -b 32 --nepoch 2000 --foldnum $foldnum --env $env --model SqueezeWB --cameras canon_eos_1D_mark3 canon_eos_600D fuji nikonD40 nikonD5200 olympus panasonic samsung sony

#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
foldnum=0
env=squeezenet_nus_stage2_fold$foldnum.`date +'%Y-%m-%d_%H-%M-%S'`
python train_stage_2.py -b 16 --nepoch 2000 --foldnum $foldnum --env $env --cameras canon_eos_1D_mark3 canon_eos_600D fuji nikonD5200 olympus panasonic samsung sony -p .pretrain/nus_stage_1.pth

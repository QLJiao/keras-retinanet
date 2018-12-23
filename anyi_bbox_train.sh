#!/bin/bash
keras_retinanet/bin/train.py --gpu="0,1" --multi-gpu=2 --multi-gpu-force --backbone=resnet152 --random-transform pascal /home/ustc/jql/VOCdevkit2007/VOC2007

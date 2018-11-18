#!/bin/bash
keras_retinanet/bin/train.py --gpu=1 --backbone=resnet152 --weights=/home/ustc/jql/x-ray/keras-retinanet/snapshots/resnet152_pascal_01_7903.h5 --random-transform pascal /home/ustc/jql/JSRT 

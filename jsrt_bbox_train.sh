#!/bin/bash
keras_retinanet/bin/train.py --gpu=0 --backbone=resnet152 --weights=/home/ustc/jql/x-ray/keras-retinanet/snapshots/model_save/anyi_resnet152_pascal_8147.h5 --random-transform pascal /home/ustc/jql/JSRT

#!/bin/bash
# keras_retinanet/bin/train.py --gpu=0 --backbone=resnet152 --random-transform pascal /home/ustc/jql/VOCdevkit2007/VOC2007
keras_retinanet/bin/train.py --gpu=0 --backbone=resnet152 --weights=/home/ustc/jql/x-ray/keras-retinanet/snapshots/model_save/anyi_NOCONCATE_resnet152_0_pascal_8098.h5 --random-transform pascal /home/ustc/jql/VOCdevkit2007/VOC2007
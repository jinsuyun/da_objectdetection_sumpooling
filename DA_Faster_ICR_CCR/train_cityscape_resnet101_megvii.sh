#!/bin/bash
save_dir="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model_res101_megvii/"
dataset="cityscape"
net="res101"
pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/resnet101_caffe.pth"

python da_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path}  --max_epochs 12

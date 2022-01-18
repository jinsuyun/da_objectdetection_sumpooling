#!/bin/bash
save_dir="/mnt/data2/JINSU/CR-DA-DET/SW_Faster_ICR_CCR/cityscape_model/model_vgg16_megvii"
dataset="cityscape"
pretrained_path="/mnt/data2/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/vgg16_caffe.pth"
net="vgg16"

CUDA_VISIBLE_DEVICES=2 python da_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex
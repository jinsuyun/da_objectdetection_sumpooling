#!/bin/bash
save_dir="/mnt/sdd/JINSU/CR-DA-DET/SW_Faster_ICR_CCR/cityscape_model/model_vgg16_Sumpooling"
dataset="cityscape"
net="vgg16"
pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/vgg16_caffe.pth"

CUDA_VISIBLE_DEVICES=0 python dafaster_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc
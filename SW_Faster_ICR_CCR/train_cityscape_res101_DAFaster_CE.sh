#!/bin/bash
save_dir="/mnt/data2/JINSU/CR-DA-DET/SW_Faster_ICR_CCR/cityscape_model/model_res101_DAF_CE"
dataset="cityscape"
net="res101"
pretrained_path="/mnt/data2/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/resnet101_caffe.pth"

CUDA_VISIBLE_DEVICES=2 python dafaster_train_net_CE.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc

#!/bin/bash
save_dir="/mnt/sdd/JINSU/CR-DA-DET/SW_Faster_ICR_CCR/cityscape_model/model_res101_DAF"
dataset="cityscape"
net="res101"
pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/resnet101_caffe.pth"

CUDA_VISIBLE_DEVICES=3 python dafaster_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc

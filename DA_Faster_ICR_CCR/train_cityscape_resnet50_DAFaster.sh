#!/bin/bash
save_dir="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model_res50_DAF"
dataset="cityscape"
net="res50"
pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/resnet50.pth"

CUDA_VISIBLE_DEVICES=0 python dafaster_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --max_epochs 12

#!/bin/bash
save_dir="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model_vgg16_megvii"
dataset="cityscape"
net="vgg16"
pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/vgg16_caffe.pth"

python da_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path}  --max_epochs 12

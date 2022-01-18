#!/bin/bash
#save_dir="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model_vgg16_DAF_sum_height_disc_cst_tr1"
#save_dir2="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model_vgg16_DAF_sum_height_disc_cst_tr2"
#save_dir3="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model_vgg16_DAF_sum_height_disc_cst_tr3"
save_dir3="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model"
dataset="cityscape"
net="vgg16"
pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pre_trained_model/vgg16_caffe.pth"

#CUDA_VISIBLE_DEVICES=2 python dafaster_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path}  --max_epochs 12 --grl #> result_txt/model_vgg16_DAF_sum_height_disc_cst_tr1.txt
#CUDA_VISIBLE_DEVICES=0 python dafaster_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir2} --pretrained_path ${pretrained_path}  --max_epochs 12 --grl > result_txt/model_vgg16_DAF_sum_height_disc_cst_tr2.txt
#CUDA_VISIBLE_DEVICES=1 python dafaster_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir3} --pretrained_path ${pretrained_path}  --max_epochs 12 --grl > result_txt/model_vgg16_DAF_sum_height_disc_cst_tr3.txt
CUDA_VISIBLE_DEVICES=3 python dafaster_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir3} --pretrained_path ${pretrained_path}  --max_epochs 12 --grl


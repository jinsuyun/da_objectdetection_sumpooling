import os
import argparse

# net = "res101"
part = "test_t"
dataset = "cityscape"
begin_epoch = 6
end_epoch = 12

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--model_prefix",
        dest="model_prefix",
        help="directory to trained model",
        default=" ",
        type=str,
    )
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="vgg16", type=str
    )

    parser.add_argument(
        "--gpus", dest="gpus", help="gpu number", default="3", type=str
    )

    args = parser.parse_args()
    return args

args = parse_args()
net=args.net
model_prefix=args.model_prefix+"/cityscape_"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
# model_prefix = "/data/experiments/DA_Faster_ICR_CCR/cityscape/model/cityscape_"
# model_prefix = "/mnt/data2/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/cityscape/model_res101_megvii/cityscape_"

if args.net=="vgg16":
    pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pretrained_model/vgg16_caffe.pth"

if args.net=="res101":
    pretrained_path="/mnt/sdd/JINSU/CR-DA-DET/DA_Faster_ICR_CCR/pretrained_model/resnet101_caffe.pth"

commond = "python eval/backup_test.py  --net {} --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)
# commond = "python eval/backup_test.py  --grl --net {} --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)
# commond = "python eval/test.py  --grl --net {} --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)
# commond = "python eval/test.py --net {} --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)
for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")
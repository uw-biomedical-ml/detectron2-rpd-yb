#!/bin/sh

#rm -rf output_dummy/
#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,5 python plain_train_net.py --config-file configs/working --num-gpus 4 SOLVER.IMS_PER_BATCH 8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,5 python plain_train_net.py --config-file configs/working --num-gpus 4 --resume SOLVER.IMS_PER_BATCH 8 MODEL.WEIGHTS model_0003599.pth


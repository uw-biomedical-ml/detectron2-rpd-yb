#!/bin/sh

#rm -rf output_dummy/
#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,5 python plain_train_net.py --config-file configs/working --num-gpus 4 SOLVER.IMS_PER_BATCH 8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python plain_train_net.py --config-file configs/working --num-gpus 7 --resume SOLVER.IMS_PER_BATCH 14 SOLVER.MAX_ITER 18000 MODEL.WEIGHTS model_0005999.pth


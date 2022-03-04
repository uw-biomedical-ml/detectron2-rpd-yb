#!/bin/sh

#rm -rf output_dummy/
#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python plain_train_net.py --config-file configs/working.yaml --num-gpus 6 SOLVER.IMS_PER_BATCH 12
#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python plain_train_net.py --config-file configs/working --num-gpus 7 --resume SOLVER.IMS_PER_BATCH 14 SOLVER.MAX_ITER 18000 MODEL.WEIGHTS model_final.pth


# # 5 fold models
# # #################################################################################################
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python plain_train_net.py \
# --config-file configs/working.yaml --num-gpus 6 SOLVER.IMS_PER_BATCH 12 \
# OUTPUT_DIR "./output_valid_fold1" \
# DATASETS.TRAIN '("fold2","fold3","fold4","fold5",)' \
# DATASETS.TEST '("fold1",)'

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python plain_train_net.py \
# --config-file configs/working.yaml --num-gpus 6 SOLVER.IMS_PER_BATCH 12 \
# OUTPUT_DIR "./output_valid_fold2" \
# DATASETS.TRAIN '("fold3","fold4","fold5","fold1",)' \
# DATASETS.TEST '("fold2",)'

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python plain_train_net.py \
# --config-file configs/working.yaml --num-gpus 6 SOLVER.IMS_PER_BATCH 12 \
# OUTPUT_DIR "./output_valid_fold3" \
# DATASETS.TRAIN '("fold4","fold5","fold1","fold2",)' \
# DATASETS.TEST '("fold3",)'

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python plain_train_net.py \
# --config-file configs/working.yaml --num-gpus 6 SOLVER.IMS_PER_BATCH 12 \
# OUTPUT_DIR "./output_valid_fold4" \
# DATASETS.TRAIN '("fold5","fold1","fold2","fold3",)' \
# DATASETS.TEST '("fold4",)'

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python plain_train_net.py \
# --config-file configs/working.yaml --num-gpus 6 SOLVER.IMS_PER_BATCH 12 \
# OUTPUT_DIR "./output_valid_fold5" \
# DATASETS.TRAIN '("fold1","fold2","fold3","fold4",)' \
# DATASETS.TEST '("fold5",)'

# #################################################################################################

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python plain_train_net.py \
--config-file configs/working.yaml --num-gpus 6 SOLVER.IMS_PER_BATCH 12 \
OUTPUT_DIR "./output_valid_test" \
DATASETS.TRAIN '("fold1","fold2","fold3","fold4","fold5",)' \
DATASETS.TEST '("test",)' \
SOLVER.MAX_ITER 12000
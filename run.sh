#!/bin/sh

rm -rf output/
python train_net.py --config-file configs/working --num-gpus 7


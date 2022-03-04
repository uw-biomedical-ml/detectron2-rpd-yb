#!/bin/sh

rm -rf results ; mkdir results
python inference.py --config-file configs/working --checkpoint output/model_0005999.pth
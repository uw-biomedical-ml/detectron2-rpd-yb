#!/usr/bin/env python


import logging
import os, glob
import torch

from detectron2.engine import DefaultPredictor
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
from tqdm import tqdm
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
import pickle
import pdb

rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_combined/"
testdir = f"{rootdir}/valid/images/all/"

th = 0.1

def grab_valid():
    return pickle.load( open( "datasets/valid_refined.pk", "rb" ) )

def grab_test():
    return pickle.load( open( "datasets/test_refined.pk", "rb" ) )

def grab_train():
    return pickle.load( open( "datasets/train_refined.pk", "rb" ) )

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    DatasetCatalog.register("rpd_valid", grab_valid)
    MetadataCatalog.get("rpd_valid").thing_classes = ["rpd"]
    cfg = setup(args)
    cfg.defrost()
    print(cfg.MODEL.WEIGHTS)
    cfg.MODEL.WEIGHTS = args.checkpoint
    print(cfg.MODEL.WEIGHTS)
    pred = DefaultPredictor(cfg)
    data = DatasetCatalog.get('rpd_valid')#grab the data annotations
    h = data[0][height]
    w = data[0][width]
    nfiles = len(data)
    #filelist = glob.glob(f"{testdir}/*.png")
    #im = cv2.imread(filelist[0])
    #h,w = np.asarray(im).shape[0:2]
    img = np.zeros((nfiles,h,w,1))
    msk = np.zeros((nfiles,h,w,2))
    out = np.zeros((nfiles,h,w,2))
    ii = 0
    for dat in tqdm(data):
        fn = dat['file_name']
        fstem = fn.split("/")[-1]
        segfn = fn.replace("/images/", "/masks/").replace("_oct", "_msk")
        gt = cv2.imread(segfn)
        im = cv2.imread(fn)
        outputs = pred(im)["instances"].to("cpu")
        filtered = outputs[outputs.scores > th]
        img[ii,:,:,0] = cv2.imread(fn,cv2.IMREAD_GRAYSCALE)
        msk[ii,:,:,0] = cv2.imread(segfn,cv2.IMREAD_GRAYSCALE)
        out[ii,:,:,0] = np.uint8(np.any(np.asarray(filtered.pred_masks),axis=0))
        #if len(filtered)==12: 
            #pdb.set_trace() 
        v = Visualizer(im, MetadataCatalog.get("rpd_valid"), scale=3.0)
        result = v.draw_instance_predictions(filtered)
        result_image = result.get_image()[:, :, ::-1]
        im_scaled = cv2.resize(im,result_image.shape[0:2][::-1],interpolation=cv2.INTER_NEAREST)
        gt_scaled = cv2.resize(gt,result_image.shape[0:2][::-1],interpolation=cv2.INTER_NEAREST)
        final = np.vstack((im_scaled, result_image, gt_scaled))
        cv2.imwrite(f"results/{fstem}", final)
        ii = ii+1
    msk[:,:,:,0] = msk[:,:,:,0]/255
    msk[:,:,:,1] = 1-msk[:,:,:,0]
    out[:,:,:,1] = 1-out[:,:,:,0]
    np.savez('./nppred/detectron_rpd_valid_vv.npz',output=out,imgs=img,masks=msk) 
    return



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


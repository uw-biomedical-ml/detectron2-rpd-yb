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

rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_combined/"
testdir = f"{rootdir}/test/images/all/"

th = 0.1

def grab_valid():
    return pickle.load( open( "datasets/valid.pk", "rb" ) )


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
    for fn in tqdm(glob.glob(f"{testdir}/*.png")):
        fstem = fn.split("/")[-1]
        segfn = fn.replace("/images/", "/masks/").replace("_oct", "_msk")
        gt = cv2.imread(segfn)
        im = cv2.imread(fn)
        outputs = pred(im)["instances"].to("cpu")
        filtered = outputs[outputs.scores > th]
        v = Visualizer(im, MetadataCatalog.get("rpd_valid"), scale=1.0)
        result = v.draw_instance_predictions(filtered)
        result_image = result.get_image()[:, :, ::-1]
        final = np.vstack((im, result_image, gt))
        cv2.imwrite(f"results/{fstem}", final)

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


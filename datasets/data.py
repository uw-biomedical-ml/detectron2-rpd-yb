#!/usr/bin/env python

from detectron2.data import DatasetCatalog
from glob import glob
import numpy as np
import pycocotools
import os
import cv2
from detectron2.structures import BoxMode
from tqdm import tqdm
import pickle

rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_combined/"

def rpd_data(grp = "train"):
    dataset = []
    instances = 0
    wrong_poly = 0

    for fn in tqdm(glob(f"{rootdir}/{grp}/images/all/*.png")):
        imageid = fn.split("/")[-1]
        segfn = fn.replace("/images/", "/masks/").replace("_oct", "_msk")
        if not os.path.isfile(segfn):
            print(fn)
        im = cv2.imread(fn)
        seg = cv2.imread(segfn)
        dat = dict(file_name = fn, height = im.shape[0], width = im.shape[1], image_id = imageid, annotations = [])
        if (np.max(seg) != 0):
            annotations = []
            seg = seg[:, :, 0]
            ret,binseg = cv2.threshold(seg, 128, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(binseg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                instances += 1
                x,y,w,h = cv2.boundingRect(c)
                epsilon = 0.001*cv2.arcLength(c,True)
                #approx = cv2.approxPolyDP(c,epsilon,True)
                #if approx.shape[0] < 6:
                #    wrong_poly += 1
                #    continue
                if len(c) < 6: 
                    wrong_poly += 1
                    continue

                anot = dict(bbox = (x,y,w,h), 
                        bbox_mode = BoxMode.XYWH_ABS, 
                        category_id = 0, 
                        segmentation = [c.flatten().tolist()])
                annotations.append(anot)
            dat["annotations"] = annotations
        dataset.append(dat)
    print(f"Found {len(dataset)} images")
    print(f"Found {instances} instances")
    print(f"Found {wrong_poly} too few vertices")
    return dataset


if __name__ == "__main__":
    for grp in ("train", "valid", "test"):
        print(grp)
        data = rpd_data(grp=grp)
        pickle.dump(data, open(f"{grp}.pk", "wb"))


#!/usr/bin/env python


from glob import glob
import os
import cv2
from .volReader import volFile
from tqdm import tqdm



script_dir = os.path.dirname(__file__)
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

import glob
import pandas as pd

def extractFiles(dataset_name, dirtoextract, extracted_path):
    if not os.path.isdir(os.path.join(extracted_path,"Extracted " + str(dataset_name))):
        print("Extracting " + dataset_name + "...")
        files_to_extract = glob.glob(os.path.join(dirtoextract,'**/*.vol'),recursive=True)
        for i,line in enumerate(tqdm(files_to_extract)):
            fpath = line.strip('\n').replace('\\','/')
            path, scan_str = fpath.strip('.vol').rsplit('/',1)
            extractpath = os.path.join(extracted_path,dataset_name,scan_str.replace('_','/'))
            os.makedirs(extractpath,exist_ok=True)
            vol = volFile(fpath)
            preffix = extractpath+'/'+scan_str+'_oct'
            vol.renderOCTscans(preffix)
    else:
        print(dataset_name + " has already been extracted.")


def rpd_data(dataset_name, extracted_path):
    dataset = []
    instances = 0
    wrong_poly = 0
    extracted_files = glob.glob(os.path.join(extracted_path,dataset_name,'**/*.png'),recursive=True)
    print("Generating dataset of images...")
    for fn in extracted_files:
        imageid = fn.split("/")[-1]
        im = cv2.imread(fn)
        dat = dict(file_name = fn, height = im.shape[0], width = im.shape[1], image_id = imageid)
        dataset.append(dat)
    print(f"Found {len(dataset)} images")
    print(f"Found {instances} instances")
    print(f"Found {wrong_poly} too few vertices")
    return dataset

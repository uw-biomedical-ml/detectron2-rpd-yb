#!/usr/bin/env python

from detectron2.data import DatasetCatalog
from glob import glob
import numpy as np
import os,sys
import cv2
from PIL import Image,ImageOps,ImageFilter,ImageSequence
from .volReader import volFile
from detectron2.structures import BoxMode
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pdb
from .yb_split_optimizers import getSplitkfold


script_dir = os.path.dirname(__file__)
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

import glob
import pandas as pd

def extractFiles(dataset_name = None, dirtoextract = None, output_path = None):
    if not os.path.isdir(os.path.join(output_path,"Extracted " + str(dataset_name))):
        print("Extracting " + dataset_name + "...")
        files_to_extract = glob.glob(os.path.join(dirtoextract,'**/*.vol'),recursive=True)
        for i,line in enumerate(tqdm(files_to_extract)):
            fpath = line.strip('\n').replace('\\','/')
            path, scan_str = fpath.strip('.vol').rsplit('/',1)
            extractpath = os.path.join(output_path,"Extracted " + str(dataset_name),scan_str.replace('_','/'))
            os.makedirs(extractpath,exist_ok=True)
            vol = volFile(fpath)
            preffix = extractpath+'/'+scan_str+'_oct'
            vol.renderOCTscans(preffix)
    else:
        print(dataset_name + " has already been extracted.")


def rpd_data(dataset_name = None, output_path = None):
    dataset = []
    instances = 0
    wrong_poly = 0
    extracted_files = glob.glob(os.path.join(output_path,"Extracted " + str(dataset_name),'**/*.png'),recursive=True)
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

#!/usr/bin/env python



import os
import cv2
from numpy import extract
from .volReader import volFile
from tqdm import tqdm
import glob
import sys
import distutils


script_dir = os.path.dirname(__file__)
class Error(Exception):
    """Base class for exceptions in this module."""
    pass



def extract_files(dirtoextract, extracted_path):
    proceed = True
    if ((os.path.isdir(extracted_path)) and (len(os.listdir(extracted_path))!=0)):
        val = input(f'{extracted_path} exists and is not empty. Files may be overwritten. Proceed with extraction? (Y/N)')
        proceed = bool(distutils.util.strtobool(val))
    if proceed:
        print(f"Extracting files from {dirtoextract} into {extracted_path}...")
        files_to_extract = glob.glob(os.path.join(dirtoextract,'**/*.vol'),recursive=True)
        for i,line in enumerate(tqdm(files_to_extract)):
            fpath = line.strip('\n')
            vol = volFile(fpath)
            fpath = fpath.replace('\\','/')
            path, scan_str = fpath.strip('.vol').rsplit('/',1)
            extractpath = os.path.join(extracted_path,scan_str.replace('_','/'))
            os.makedirs(extractpath,exist_ok=True)
            preffix = os.path.join(extractpath, scan_str+'_oct')
            vol.renderOCTscans(preffix)
    else:
        pass


def rpd_data(extracted_path):
    dataset = []
    instances = 0
    wrong_poly = 0
    extracted_files = glob.glob(os.path.join(extracted_path,'**/*.png'),recursive=True)
    print("Generating dataset of images...")
    for fn in extracted_files:
        fn_adjusted = fn.replace('\\','/')
        imageid = fn_adjusted.split("/")[-1]
        im = cv2.imread(fn)
        dat = dict(file_name = fn_adjusted, height = im.shape[0], width = im.shape[1], image_id = imageid)
        dataset.append(dat)
    print(f"Found {len(dataset)} images")
    print(f"Found {instances} instances")
    print(f"Found {wrong_poly} too few vertices")
    return dataset

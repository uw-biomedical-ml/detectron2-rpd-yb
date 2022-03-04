from detectron2.data import DatasetCatalog
import glob
import numpy as np
import os,sys
import cv2
from PIL import Image,ImageOps,ImageFilter,ImageSequence
from detectron2.structures import BoxMode
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pdb

rpath='/data/ssong/rpd_data' #root path for files
dirtofiles = rpath +'/Test' #extracted from 

def main():
    data = []
    files_to_extract = glob.glob(os.path.join(dirtofiles,'*.vol'),recursive=True)
    # print(files_to_extract)
    for i,line in enumerate(tqdm(files_to_extract)):
        fpath = line.strip('\n').replace('\\','/')
        # print(fpath)
        path, scan_str = fpath.strip('.vol').rsplit('/',1)
        scan_str = scan_str.split('_')
        scan_str.append(fpath)
        data.append(scan_str)
    df = pd.DataFrame(data, columns = ['ptid', 'eye','path'])
    df.to_csv('test.csv', index = False)

if __name__ == "__main__":
    main()
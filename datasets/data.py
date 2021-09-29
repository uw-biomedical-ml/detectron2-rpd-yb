#!/usr/bin/env python

from detectron2.data import DatasetCatalog
from glob import glob
import numpy as np
import os
import cv2
from detectron2.structures import BoxMode
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pdb


script_dir = os.path.dirname(__file__)
# rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_combined_folds/"
#rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_101001_OS"
# rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_val"
rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_training_val_folds"

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

import glob
import pandas as pd
def checkSplit2(path):
    """Check for overlap across folds.

    Args:
        path (str): Parent director of fold subdirectories. 
        The fold subdirectories are assumed to have structure ./fold/images/all/ and the name of the file is assumed to start with the group id.

    Returns:
        df (pd.DataFrame): A dataframe equivalent of the intersection matrix of folds.
    """
    dirs = glob.glob(path+'/*/')
    splitdict = {}
    for dir in dirs:
        splitname = dir.strip('/').split('/')[-1]
        files = os.listdir(dir+'images/all/') 
        ptids = set([f.split('_')[0] for f in files])
        splitdict[splitname]=ptids
    df = pd.DataFrame(index = splitdict.keys(),columns=splitdict.keys())
    for key,value in splitdict.items():
        for key2,value2 in splitdict.items():
            df.loc[key,key2]= (len(set.intersection(value,value2)))
    return df

def checkHandler(df):
    if (df.values.sum() - np.diag(df.values).sum())>0:
        raise Error('There are overlapping folds!')

def findEdgeIndex(df,thbool,idx):
    """Find the first location at which a threshold is met after a given index.

    Args:
        df (pandas.core.frame.DataFrame): DataFrame with column 'y' being integral of vertical pixel values, index is horizontal pixel position 'x'
        thbool (DataFrame): DataFrame of booleans indicating y > some threshold
        idx (int): The Index of the df to start from

    Returns:
        int: The index of the first edge where threshold is surpassed or None if no edge is found.
    """
    a = df[thbool].loc[idx:]
    if len(a)>0:
         return a.index[0]
    else:
        return None

def findSegments(df,thresh=0,minpixels=15):
    """Find all the regions where a threshold is met.

    Args:
        df (pandas.core.frame.DataFrame): DataFrame with column 'y' being integral of vertical pixel values, index is horizontal pixel position 'x'
        thresh (int, optional): Integral pixel threshold above which region is considered to be a segment. Defaults to 0 (any non-zero pixel value).

    Returns:
        list of tuples: List of tuple indexes corresponding to start and stop of segment.
    """

    nonzero = df['y']>thresh #boolean array
    segs = []
    idx1=0
    idx2=0
    while (idx1!=None): #until the end of the line
        #find rising edge
        idx1 = findEdgeIndex(df,nonzero,idx2)
        if idx1==None:
            break
        #find falling edge
        idx2 = findEdgeIndex(df,~nonzero,idx1)
        if idx2==None:
            break
        if (idx2-idx1)<minpixels: #if bump less than minpixels in width,skip
            continue
        segs.append((idx1,idx2))
    return segs

def findBumps(df,segs,thresh=3):
    """Find instances within continous segments based on threshold.

    Args:
        df (pandas.core.frame.DataFrame): DataFrame with column 'y' being integral of vertical pixel values, index is horizontal pixel position 'x'
        segs (list of tuples): List of tuple indexes corresponding to start and stop of continuous segments. 
        thresh (int, optional): Integral pixel threshold above which regions is considered to be an instance. Defaults to 3 pixels.

    Returns:
        list of lists of tuples: Outer list corresponds to continious segments, inner list contains indexes corresponding to start and stop of instances within segments.
    """
    bumps = []
    for seg in segs:
        bumps.append(findSegments(df.loc[seg[0]:seg[1]],thresh))
    return bumps

def findBoundaries(df, segs): 
    """Find locations of minima between instances within segments.

    Args:
        df (pandas.core.frame.DataFrame): DataFrame with column 'y' being integral of vertical pixel values, index is horizontal pixel position 'x'
        segs (list of lists of tuples): Outer list corresponds to continious segments, inner list contains indexes corresponding to start and stop of instances within segments.

    Returns:
        list of indexes: List of indexes corresponding to minima between instances of segments.
    """
    idx = []
    for seg in segs:
        for i in range(len(seg)-1):
            dfseg = df.loc[seg[i][1]:seg[i+1][0]]
            ix = np.median(dfseg[dfseg == dfseg.min()].dropna().index)
            idx.append(int(ix))
    return idx

def height_crop_image(im,msk,height_target=256):
    yhist = im.sum(axis=1) #integrate over width of image
    mu = np.average(np.arange(yhist.shape[0]),weights = yhist)
    h1 = int(np.floor(mu-height_target/2))
    h2 = int(np.ceil(mu+height_target/2))
    return im[h1:h2,:],msk[h1:h2,:]

def visualize_breakup(im,binseg,segs,bumps,idx):
    """Plot segment and instance indicators overlayed on top of image of original mask.

    Args:
        im (numpy array): image of mask
        segs (list of tuples): List of tuple indexes corresponding to start and stop of continuous segments.
        bumps (list of lists of tuples): Outer list corresponds to continious segments, inner list contains indexes corresponding to start and stop of instances within segments.
        idx (list of indexes): List of indexes corresponding to minima between instances of segments.

    Returns:
        figure, axes objects: Segment and instance indicators overlayed on top of image of original mask.
    """
    im,binseg = height_crop_image(im,binseg)
    fig,ax = plt.subplots(2,1,figsize=[10,8],dpi=300)
    ax[0].imshow(im)
    ax[1].imshow(binseg)
    ind = binseg.sum(axis=1).nonzero()[0].max()
    for i,seg in enumerate(segs):      
        ax[1].hlines(ind+30,seg[0],seg[1],color='green')
        for bump in bumps[i]:
            ax[1].hlines(ind+50,bump[0],bump[1],color='pink')
    ax[1].vlines(idx,ind+50,im.shape[0],color='red')
    return fig,ax


def rpd_data(grp = "train",data_has_ann=True):
    dataset = []
    instances = 0
    wrong_poly = 0
    ii=0
    outname = os.path.join(script_dir, grp+'_instance_refine_all.pdf')
    with PdfPages(outname) as pdf:
        for fn in tqdm(glob(f"{rootdir}/{grp}/images/all/*.png")):
            imageid = fn.split("/")[-1]
            segfn = fn.replace("/images/", "/masks/").replace("_oct", "_msk")
            if not os.path.isfile(segfn):
                print(fn)
            im = cv2.imread(fn)
            seg = cv2.imread(segfn)
            dat = dict(file_name = fn, height = im.shape[0], width = im.shape[1], image_id = imageid)
            if data_has_ann: 
                annotations = []             
                if (np.max(seg) != 0):
                    
                    seg = seg[:, :, 0]
                    ret,binseg = cv2.threshold(seg, 128, 255, cv2.THRESH_BINARY)
                    #integral of segmentation
                    y = (binseg/binseg.max()).sum(axis=0).astype(int) 
                    df = pd.DataFrame(y,columns=['y'])
                    #find and break up segments into instances
                    segs = findSegments(df)
                    bumps = findBumps(df,segs,thresh=3)
                    idx = findBoundaries(df,bumps)
                    #insert breaks in image
                    for i in idx:
                        binseg[:,i]=0
                    if len(idx)>0:
                        fig,ax = visualize_breakup(im[:,:,0],binseg,segs,bumps,idx)
                        pdf.savefig(fig)
                        plt.close(fig)
                    #find contours and bounding boxes
                    _, contours, hierarchy = cv2.findContours(binseg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        instances += 1
                        x,y,w,h = cv2.boundingRect(c)
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
            ii=ii+1
#             if ii==500:
#                 break
    print(f"Found {len(dataset)} images")
    print(f"Found {instances} instances")
    print(f"Found {wrong_poly} too few vertices")
    return dataset


if __name__ == "__main__":
    dfcheck = checkSplit2(rootdir)
    checkHandler(dfcheck)
    for grp in ("fold1", "fold2", "fold3", "fold4","fold5","test"):
        print(grp)
        data = rpd_data(grp=grp,data_has_ann=True)
        pickle.dump(data, open(os.path.join(script_dir,f"{grp}_refined.pk"), "wb"))
        #pickle.dump(data, open(f"val_refined.pk", "wb"))



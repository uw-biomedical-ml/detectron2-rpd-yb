#!/usr/bin/env python

from detectron2.data import DatasetCatalog
from glob import glob
import numpy as np
import os,sys
import cv2
from PIL import Image,ImageOps,ImageFilter,ImageSequence
from volReader import volFile
from detectron2.structures import BoxMode
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pdb
from yb_split_optimizers import getSplitkfold


script_dir = os.path.dirname(__file__)
# rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_combined_folds/"
#rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_101001_OS"
# rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_RPDHimeesh_val"
#rootdir = "/data/amd-data/cera-rpd/cera-rpd-train/data_training_val_folds"

rpath='/data/amd-data/cera-rpd' #root path for files
dirtoextract = rpath +'/Test' #extracted from 
filedir = rpath +'/Test_extracted' #split from

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

import glob
import pandas as pd

def inithval_list():
    cmap = {
            'yellow': [255,255,0],
            'green': [0,255,0],
            'red': [255, 0, 0],
            'black': [0,0,0],
            'white': [255,255,255],
            'gray' : [7, 7, 7]
            
            }
    
    hval_list = [cmap['yellow'],cmap['white'],cmap['red'],cmap['black']] #mapping
    p_list = [1.0, 0.8, 0.5, 0]
    return hval_list,p_list

def extractFiles(masks_exist=True):
    files_to_extract = glob.glob(os.path.join(dirtoextract,'*.vol'),recursive=True)
    for i,line in enumerate(tqdm(files_to_extract)):
        fpath = line.strip('\n')
        path, scan_str = fpath.strip('.vol').rsplit('/',1)
        extractpath = path + '_extracted/'+scan_str
        os.makedirs(extractpath,exist_ok=True)
        vol = volFile(fpath)
        #myhash = get_random_string(6,allowed_chars=ascii_uppercase+digits)
        preffix = extractpath+'/'+scan_str+'_oct'
        #print('\n'+ preffix)
        vol.renderOCTscans(preffix)

 #        mfile = path+'/'+(scan_str.rsplit('_',1)[0])+'.tiff'
        if (masks_exist):
            mfile = path+'/'+scan_str+'.tiff'
            msk = Image.open(mfile)
            for i,page in enumerate(ImageSequence.Iterator(msk)):
                page.save(extractpath+'/'+scan_str+'_msk-{:03d}.png'.format(i))                
        else: #create blank mask
            page = Image.new('RGB',(1024,496)) #default black image
            for i in range(vol.oct.shape[0]):
                page.save(extractpath+'/'+scan_str+'_msk-{:03d}.png'.format(i))

def countColors(msk,hval_list):
    '''For PIL RGB image msk, return the number of pixels of each color specified in in hval_list.'''
    color_list = msk.getcolors()
    unzipped_list = list(zip(*color_list))
    #detect colors
    
    
    cnt = np.zeros((len(hval_list)+1,)) #last entry is for background

    for i in range(len(unzipped_list[1])):#for all unordered colors detected in image
        if list(unzipped_list[1][i]) in hval_list: #if color present in hval_list
            idx = hval_list.index(list(unzipped_list[1][i])) #find index
            cnt[idx]=unzipped_list[0][i] #and assign to count
    cnt[-1] = msk.width*msk.height - cnt[:-1].sum() #rest are background pixels
    return cnt 

def pruneMapColors(im,hval_list):
    '''
    Parameters
    ----------
    im : image as numpy array.
    hval_list : list of rgb tuples corresponding to ordinal classes (excluding background, first is furthest from background)

    Returns
    -------
    newim : copy of original image but only including colors in hval_list


    '''
    newim = np.zeros(im.shape)
    
    for i,color in enumerate(hval_list):
        try:
            indices = np.where(np.all(im == color,axis=2))
        except:
            print(im.shape)
        newim[indices[0],indices[1]]=color
       
    return newim

def convertMapToGrayScale(im,hval_list,p_list):
    '''
    Converts image to grayscale, assigning pixel value in p_list to corresponding color in hval_list.

    Parameters
    ----------
    im : numpy array
        image.
    hval_list : 
        list of rgb values.
    p_list : 
        list of probabilities.

    Returns
    -------
    newim : numpy array
        converted image.

    '''
    im2 = im.reshape(-1,3)
    newim = np.zeros((im2.shape[0],1))
 
    for i in range(im2.shape[0]):
        if list(im2[i,:]) in hval_list:
            idx = hval_list.index(list(im2[i,:]))
            newim[i,0] = p_list[idx]*255
    
    
    newim = newim.reshape(im.shape[0],im.shape[1])
    return newim

def createDf():
    '''Create dataframe of images contained in filelist. Include colored pixel count.'''
    val_list,p_list = inithval_list() 
    df = pd.DataFrame(columns=['ptid','eye','scan','img_path','msk_path','yellow','white','red','black'])
    filelist = glob.glob(os.path.join(filedir,'*/*oct*.png'),recursive=True)
    for i,line in enumerate(tqdm(filelist)):
        fpath = line.strip('\n')
        # pdb.set_trace()
        path, scan_str = fpath.strip('.png').rsplit('/',1)
        mskfile = path + '/' + scan_str.replace('oct','msk') +'.png'
        data = path.split('/')[-1].split('_')+[scan_str] +[fpath] + [mskfile]

        #read image
        if os.path.exists(mskfile):
            msk= Image.open(mskfile)
        else:
            print('Error: {} not found!'.format(mskfile))
            sys.exit()
            
        if (msk.mode !='RGBA' and msk.mode != 'RGB' and msk.mode != 'P'):
            print('Error: image mask {} is mode {}, not RGB or RGBA.'.format(mskfile,msk.mode))
            sys.exit()
        else:
            msk = msk.convert('RGB')
            
        cnt_arr = countColors(msk,val_list[:-1])
        df.loc[i] = data + list(cnt_arr)
    #save data frame
    df.to_csv(filedir + '/dataframe.csv',index=False)
    return df

def process_masks(df,mode='RGB',binary_classes = 2):
    '''
    Generates processed masks according to mode.

    Parameters
    ----------
    df : A dataframe containing the data paths.
    mode : The default is 'RGB'.
        'RGB' generates RBG masks according to hval_list (inside function).
        'gray' generates grayscale masks, mapping the colors in hval_list to p_list*255
        'sanity' generates RBG masks and grayscale images from the masks generated with 'gray'. These are images to test a multiclass model with.
        'binary' generates binary masks for binary classification.
    binary_classes : The top classes to convert to positives from hval_list. The default is 2.

    Returns
    -------
    df_processed : A dataframe containing the paths of the processed data.

    '''
    
    assert (mode=='gray')|(mode=='RGB')|(mode=='sanity')|(mode=='binary'),'Invalid entry for "mode". Valid values are "RBG", "gray", "binary" or "sanity".'
    hval_list,p_list = inithval_list() #RGB values and corresponding probabilities
    # num_classes = len(hval_list)

    #check the split from the directories
    dfcheck = checkSplit2(df)
    print(dfcheck)
    if (dfcheck.values.sum() - np.diag(dfcheck.values).sum())>0:
        raise Error('There are overlapping folds!')
    
    for img_path,msk_path in tqdm(df[['img_path','msk_path']].values):
        
        #convert mask
        if os.path.exists(msk_path):
            msk= Image.open(msk_path)
        else:
            print('Error: {} not found!'.format(msk_path))
            sys.exit()
            
        if (msk.mode !='RGBA' and msk.mode != 'RGB' and msk.mode != 'P'):
            print('Error: image mask {} is mode {}, not RGB or RGBA.'.format(msk_path,msk.mode))
            sys.exit()
        else:
            msk = np.array(msk.convert('RGB'))

        msk = pruneMapColors(msk, hval_list[:-1])
        
        if (mode == 'gray')|(mode == 'sanity')|(mode == 'binary'):
            if mode == 'binary':
                newmsk = convertMapToGrayScale(msk,hval_list[:binary_classes],[1.]*binary_classes)
            else:
                newmsk = convertMapToGrayScale(msk,hval_list[:-1],p_list) #gray mode
            newmsk = Image.fromarray(newmsk.astype('uint8'),mode='L')
            if mode=='sanity': #save grayscale mask as image
                newmsk.save(img_path.replace('oct','oct-sanity'))    
        if ((mode == 'RGB')| (mode =='sanity')): #RGB mask
            newmsk = Image.fromarray(msk.astype('uint8'),mode='RGB')
        newmsk.save(msk_path.replace('msk','msk-'+ mode))   

    dfprocessed = df.assign(msk_path = [v.replace('msk','msk-'+mode) for v in df.msk_path.values])
    dfprocessed.to_csv(filedir + '/dataframe_processed.csv',index=False)
    return dfprocessed


def checkSplit2(df):
    """Check for overlap across folds in df.

    Args:
        df (pd.DataFrame): specifies fold for each image. 

    Returns:
        dffolds (pd.DataFrame): A dataframe equivalent of the intersection matrix of folds.
    """
    dfsets = df.groupby('fold')['ptid'].apply(lambda x :set(x))
    splitdict = {}
    dffolds = pd.DataFrame(index = dfsets.index,columns=dfsets.index)
    for key,value in dfsets.items():
        for key2,value2 in dfsets.items():
            dffolds.loc[key,key2]= (len(set.intersection(value,value2)))
    return dffolds

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


def rpd_data(df, grp = "train",data_has_ann=True):
    df = df[df.fold==grp]
    dataset = []
    instances = 0
    wrong_poly = 0
    ii=0
    outname = os.path.join(script_dir, grp+'_instance_refine_all.pdf')
    with PdfPages(outname) as pdf:
        for fn,segfn in tqdm(df[['img_path','msk_path']].values):
        #for fn in tqdm(glob(f"{rootdir}/{grp}/images/all/*.png")):
            imageid = fn.split("/")[-1]
            #segfn = fn.replace("/images/", "/masks/").replace("_oct", "_msk")
            # if not os.path.isfile(segfn):
            #     print(fn)
            im = cv2.imread(fn)
            
            dat = dict(file_name = fn, height = im.shape[0], width = im.shape[1], image_id = imageid)
            if data_has_ann: 
                seg = cv2.imread(segfn)    
                annotations = []             
                if (np.max(seg) != 0):
                    
                    seg = seg[:, :, 0]
                    ret,binseg = cv2.threshold(seg, 128, 255, cv2.THRESH_BINARY) #sourch seg is grayscale
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
                    #pdb.set_trace()
                    contours, hierarchy = cv2.findContours(binseg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


# if __name__ == "__main__":
#     dfcheck = checkSplit2(rootdir)
#     checkHandler(dfcheck)
#     for grp in ("fold1", "fold2", "fold3", "fold4","fold5","test"):
#         print(grp)
#         data = rpd_data(grp=grp,data_has_ann=True)
#         pickle.dump(data, open(os.path.join(script_dir,f"{grp}_refined.pk"), "wb"))
#         #pickle.dump(data, open(f"val_refined.pk", "wb"))



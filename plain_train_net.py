#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.
This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
from fvcore.common.checkpoint import Checkpointer
import torch
from torch.nn.parallel import DistributedDataParallel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval,Params
from pycocotools.mask import decode

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
import detectron2.data.transforms as T
from detectron2.structures import Instances

from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
import pickle
import numpy as np
import pandas as pd
import pdb
import warnings
from tqdm import tqdm

logger = logging.getLogger("detectron2")
warnings.filterwarnings("ignore",category=UserWarning)

def grab_dataset(name):
    def f():
        return pickle.load( open( "datasets/"+name+"_refined.pk", "rb" ) )
    return f   

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
import cv2
from PIL import Image
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ybpres.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
import json
import sys


def checkSplits(dataset_list):
    splitdict={}
    for name in dataset_list:
        dat = grab_dataset(name)()
        ptid_set = set([d['image_id'].split('_')[0] for d in dat])
        splitdict[name] = ptid_set
    dfcheck = pd.DataFrame(index = splitdict.keys(),columns=splitdict.keys())
    for key,value in splitdict.items():
        for key2,value2 in splitdict.items():
            dfcheck.loc[key,key2]= (len(set.intersection(value,value2)))
    return dfcheck       
class OutputVis():

    def __init__(self,dataset_name,cfg=None,prob_thresh=0.5,pred_mode='model',pred_file=None,has_annotations=True, draw_mode = 'default'):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.prob_thresh = prob_thresh
        self.data = DatasetCatalog.get(dataset_name)
        if pred_mode =='model':
            self.predictor = DefaultPredictor(cfg)
            self._mode = 'model'
        elif pred_mode =='file':
            with open(pred_file,'r') as f:
                self.pred_instances = json.load(f)
            self.instance_img_list = [p['image_id'] for p in self.pred_instances]
            self._mode = 'file'
        else:
            sys.exit('Invalid mode. Only "model" or "file" permitted.')
        self.has_annotations = has_annotations
        permitted_draw_modes = ['default','bw']
        if draw_mode not in permitted_draw_modes:
            sys.exit('draw_mode must be one of the following: {}'.format(permitted_draw_modes))
        self.draw_mode = draw_mode

    def get_ori_image(self,ImgId):
        dat = self.get_gt_image_data(ImgId) #gt
        im = cv2.imread(dat['file_name']) #input to model
        v_gt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=3.0)
        result_image = v_gt.output.get_image() #get original image
        img = Image.fromarray(result_image)
        return img

    def get_gt_image_data(self,ImgId):
        gt_data = next(item for item in self.data if (item['image_id'] == ImgId))
        return gt_data    

    def produce_gt_image(self,dat,im):
        v_gt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=3.0) 
        if (self.has_annotations): #ground truth boxes and masks
            segs = [ddict['segmentation'] for ddict in dat['annotations']]
            if self.draw_mode is 'bw':
                BBoxes = None
                assigned_colors = ['r']*len(segs)
            else: #default behavior
                bboxes = [ddict['bbox'] for ddict in dat['annotations']]
                BBoxes = detectron2.structures.Boxes(bboxes)
                BBoxes = detectron2.structures.BoxMode.convert(BBoxes.tensor,from_mode=1,to_mode=0) #0= XYXY, 1 = XYWH
                assigned_colors = None
            
            result_image = v_gt.overlay_instances(boxes=BBoxes,masks=segs,assigned_colors=assigned_colors, alpha=1.0).get_image()
        else:
            result_image = v_gt.output.get_image() #get original image if no annotations
        img = Image.fromarray(result_image)
        return img

    def produce_model_image(self,ImgId,dat,im):
        v_dt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=3.0)
        v_dt._default_font_size = 14

        #get predictions from model or file
        if self._mode=='model':
            outputs = self.predictor(im)["instances"].to("cpu")
        elif self._mode=='file':
            outputs = self.get_outputs_from_file(ImgId,(dat['height'],dat['width']))  
        outputs = outputs[outputs.scores>self.prob_thresh] #apply probability threshold to instances
        if self.draw_mode is 'bw':
            result_model = v_dt.overlay_instances(masks=outputs.pred_masks,assigned_colors=['r']*len(outputs), alpha=1.0).get_image()
        else: #default behavior
            result_model = v_dt.draw_instance_predictions(outputs).get_image()    
        img_model = Image.fromarray(result_model)
        return img_model

    def get_image(self,ImgId): 
        dat = self.get_gt_image_data(ImgId) #gt
        im = cv2.imread(dat['file_name']) #input to model
        img = self.produce_gt_image(dat,im)
        img_model = self.produce_model_image(ImgId,dat,im)
        return img, img_model
  
    def get_outputs_from_file(self,ImgId,imgsize):       
        pred_boxes = []
        scores = []
        pred_classes = []
        pred_masks = []
        for i,img in enumerate(self.instance_img_list):
            if img==ImgId:
                pred_boxes.append(self.pred_instances[i]['bbox'])
                scores.append(self.pred_instances[i]['score'])
                pred_classes.append(int(self.pred_instances[i]['category_id']))
                #pred_masks_rle.append(self.pred_instances[i]['segmentation'])
                pred_masks.append(decode(self.pred_instances[i]['segmentation']))
        BBoxes = detectron2.structures.Boxes(pred_boxes)
        pred_boxes = detectron2.structures.BoxMode.convert(BBoxes.tensor,from_mode=1,to_mode=0) #0= XYXY, 1 = XYWH
        inst_dict = dict(pred_boxes = pred_boxes,scores=torch.tensor(np.array(scores)),pred_classes=torch.tensor(np.array(pred_classes)),pred_masks = torch.tensor(np.array(pred_masks)).to(torch.bool))#pred_masks_rle=pred_masks_rle)
        outputs = detectron2.structures.Instances(imgsize,**inst_dict)
        return outputs

    @staticmethod
    def height_crop_range(im,height_target=256):
        yhist = im.sum(axis=1) #integrate over width of image
        mu = np.average(np.arange(yhist.shape[0]),weights = yhist)
        h1 = int(np.floor(mu-height_target/2))
        h2 = int(np.ceil(mu+height_target/2))
        return range(h1,h2)

    def output_to_pdf(self,ImgIds,outname,dfimg=None):

        gtstr = ''
        dtstr = ''
        
        if dfimg is not None:
            gtcols = dfimg.columns[['gt_' in col for col in dfimg.columns]]
            dtcols = dfimg.columns[['dt_' in col for col in dfimg.columns]]
            
        with PdfPages(outname) as pdf:
            for imgid in tqdm(ImgIds):
                img, img_model = self.get_image(imgid)
                #pdb.set_trace()
                crop_range = self.height_crop_range(np.array(img.convert('L')),height_target=256*3)
                img = np.array(img)[crop_range]
                img_model = np.array(img_model)[crop_range]
 
                fig, ax = plt.subplots(2,1,figsize=[22,10],dpi=200)
                ax[0].imshow(img)
                ax[0].set_title(imgid+' Ground Truth')
                ax[0].set_axis_off()
                ax[1].imshow(img_model)
                ax[1].set_title(imgid+' Model Prediction')
                ax[1].set_axis_off()
                if dfimg is not None: #annotate with provided stats
                    gtstr = ['{:s}={:.2f}'.format(col,dfimg.loc[imgid,col]) for col in gtcols]
                    ax[0].text(0,.05*(ax[0].get_ylim()[0]),gtstr,color='white',fontsize=14)
                    dtstr = ['{:s}={:.2f}'.format(col,dfimg.loc[imgid,col]) for col in dtcols]                
                    ax[1].text(0,.05*(ax[1].get_ylim()[0]),dtstr,color='white',fontsize=14)                   
                pdf.savefig(fig)
                plt.close(fig)

    def save_imgarr_to_tiff(self,imgs,outname):
        if len(imgs) > 1:
            imgs[0].save(outname, tags = "", compression = "tiff_deflate", save_all=True, append_images=imgs[1:])
        else:
            imgs[0].save(outname) 

    def output_ori_to_tiff(self,ImgIds,outname):
        imgs = [] 
        for imgid in tqdm(ImgIds):
            img_ori = self.get_ori_image(imgid) #PIL Image
            imgs.append(img_ori)
        self.save_imgarr_to_tiff(imgs,outname)

    def output_pred_to_tiff(self,ImgIds,outname):
        imgs = [] 
        for imgid in tqdm(ImgIds):
            dat = self.get_gt_image_data(imgid) #gt
            im = cv2.imread(dat['file_name']) #input to model
            img_dt = self.produce_model_image(imgid,dat,im)
            imgs.append(img_dt)
        self.save_imgarr_to_tiff(imgs,outname)

    def output_all_to_tiff(self,ImgIds,outname):
        imgs = []
        for imgid in tqdm(ImgIds):
            img_gt, img_dt = self.get_image(imgid)
            img_ori = self.get_ori_image(imgid)
            hcrange = list(self.height_crop_range(np.array(img_ori.convert('L')),height_target=256*3))
            img_result = Image.fromarray(np.concatenate((np.array(img_ori.convert('RGB'))[hcrange,:],np.array(img_gt)[hcrange,:],np.array(img_dt)[hcrange])))
            imgs.append(img_result)
        self.save_imgarr_to_tiff(imgs,outname)


#########################################################################################################
    def output_masks_to_tiff(self, ImgIds, ptid, eye):
        imgs = []
        for index in range(len(ImgIds)):
            gt_data = next(item for item in self.data if (item['image_id'] == ImgIds[index]))   
            dat = gt_data
            blank = Image.new('RGB',(dat['width'],dat['height'])) #default black image
            v_dt = Visualizer(blank, MetadataCatalog.get(self.dataset_name), scale=3.0)
            v_dt._default_font_size = 14
            outputs = self.get_outputs_from_file(ImgIds[index],(dat['height'],dat['width']))
            outputs = outputs[outputs.scores>self.prob_thresh]
            result_model = v_dt.overlay_instances(masks=outputs.pred_masks,assigned_colors=['w']*len(outputs), alpha=1.0).get_image()
            pil_model = Image.fromarray(result_model)
            imgs.append(pil_model)
        if not os.path.isdir('extracted_test'):
            os.mkdir('extracted_test')
        if len(imgs) > 1:
            imgs[0].save("extracted_test/" + str(ptid) + "_" + str(eye) + "-bmasks.tif", tags = "test", compression = "tiff_deflate", save_all=True, append_images=imgs[1:])
        else:
            imgs[0].save("extracted_test/" + str(ptid) + "_" + str(eye) + "-bmasks.png")
    
    def output_overlay_masks_to_tiff(self, ImgIds, ptid, eye):
        imgs = []
        for index in range(len(ImgIds)):
            gt_data = next(item for item in self.data if (item['image_id'] == ImgIds[index]))   
            dat = gt_data
            im = cv2.imread(dat['file_name']) #input to model
            v_dt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=3.0)
            v_dt._default_font_size = 14
            outputs = self.get_outputs_from_file(ImgIds[index],(dat['height'],dat['width']))
            outputs = outputs[outputs.scores>self.prob_thresh]
            result_model = v_dt.overlay_instances(masks=outputs.pred_masks,assigned_colors=['r']*len(outputs), alpha=1.0).get_image()
            pil_model = Image.fromarray(result_model)
            imgs.append(pil_model)
        if not os.path.isdir('extracted_test_overlays'):
            os.mkdir('extracted_test_overlays')
        if len(imgs) > 1:
            imgs[0].save("extracted_test_overlays/" + str(ptid) + "_" + str(eye) + "-bmasks_overlay.tif", tags = "test", compression = "tiff_deflate", save_all=True, append_images=imgs[1:])
        else:
            imgs[0].save("extracted_test_overlays/" + str(ptid) + "_" + str(eye) + "-bmasks_overlay.png")

    def output_instances_masks_to_tiff(self, ImgIds, ptid, eye):
        imgs = []
        for index in range(len(ImgIds)):
            gt_data = next(item for item in self.data if (item['image_id'] == ImgIds[index]))   
            dat = gt_data
            im = cv2.imread(dat['file_name']) #input to model
            v_dt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=3.0)
            v_dt._default_font_size = 14
            outputs = self.get_outputs_from_file(ImgIds[index],(dat['height'],dat['width']))
            outputs = outputs[outputs.scores>self.prob_thresh]
            result_model = v_dt.draw_instance_predictions(outputs).get_image()
            pil_model = Image.fromarray(result_model)
            imgs.append(pil_model)
        if not os.path.isdir('instances_mask_overlays'):
            os.mkdir('instances_mask_overlays')
        if len(imgs) > 1:
            imgs[0].save("instances_mask_overlays/" + str(ptid) + "_" + str(eye) + "-ipmasks_overlay.tif", tags = "test", compression = "tiff_deflate", save_all=True, append_images=imgs[1:])
        else:
            imgs[0].save("instances_mask_overlays/" + str(ptid) + "_" + str(eye) + "-ipmasks_overlay.png")

#########################################################################################################
class EvaluateClass(COCOEvaluator):  
    def __init__(self,dataset_name, output_dir,prob_thresh=0.5,iou_thresh = 0.1,evalsuper=True):
        super().__init__(dataset_name,tasks={'bbox','segm'},output_dir = output_dir)
        self.dataset_name = dataset_name
        self.mycoco=None #pycocotools.cocoEval instance
        self.cocoDt=None
        self.cocoGt=None
        self.evalsuper = evalsuper # if True, run COCOEvaluator.evaluate() when self.evaluate is run
        self.prob_thresh = prob_thresh #instance probabilty threshold for scalars (precision,recall,fpr for scans)
        self.iou_thresh = iou_thresh #iou threshold for defining precision,recall
        self.pr = None
        self.rc = None
        self.fpr = None
    def reset(self): 
        super().reset()
        self.mycoco=None
    def process(self, inputs, outputs):
        super().process(inputs,outputs)
    def evaluate(self):
        #with nostdout(): #suppress the coco summarize statment (the one with APs)
        if self.evalsuper:
            _ = super().evaluate() #this call populates coco_instances_results.json
        comm.synchronize()
        if not comm.is_main_process():
            return ()
        self.cocoGt = COCO(os.path.join(self._output_dir,self.dataset_name +'_coco_format.json')) #produced when super is initialized
        self.cocoDt = self.cocoGt.loadRes(os.path.join(self._output_dir,'coco_instances_results.json')) #load detector results
        self.mycoco = COCOeval(self.cocoGt,self.cocoDt,iouType ='segm')
        self.num_images = len(self.mycoco.params.imgIds)
        print('Calculated metrics for {} images'.format(self.num_images))
        self.mycoco.params.iouThrs = np.arange(.10,.6,.1)
        #mycoco.params.recThrs = [0,.5,.75,1]
        self.mycoco.params.maxDets = [100]
        #mycoco.params.imgIds=[imgId]
        self.mycoco.params.areaRng = [[0, 10000000000.0]]

        self.mycoco.evaluate()
        self.mycoco.accumulate()

        self.pr = self.mycoco.eval['precision'][:, #iouthresh
                         :, #recall level
                         0, #catagory
                         0, #area range
                         0] #max detections per image
        self.rc = self.mycoco.params.recThrs
        self.iou = self.mycoco.params.iouThrs
        self.scores = self.mycoco.eval['scores'][:,:,0,0,0] #unreliable if GT has no instances
        p,r = self.get_precision_recall()
        return p,r

    def plot_PRcurve(self,ax=None):
        if ax==None:
            fig, ax = plt.subplots(1,1)
        for i in range(len(self.iou)):
            ax.plot(self.rc,self.pr[i],label = '{:.2}'.format(self.iou[i]))
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('')
        ax.legend(title='IoU')
        
    def plot_recall_vs_prob(self):
        plt.figure()
        for i in range(len(self.iou)):
            plt.plot(self.rc,self.scores[i],label = '{:.2}'.format(self.iou[i]))
        plt.ylabel('Model probability')
        plt.xlabel('Recall')
        plt.legend(title='IoU')
            
    def get_precision_recall(self):
        iou_idx,rc_idx = self._find_iou_rc_inds()
        precision = self.pr[iou_idx,rc_idx]
        recall = self.rc[rc_idx]
        return precision,recall

    def _calculate_fpr_matrix(self):
        #FP rate, 1 RPD in image = FP
        if (self.scores.min()==-1) and (self.scores.max()==-1):
            print('WARNING: Scores for all iou thresholds and all recall levels are not defined. This can arise if ground truth annotations contain no instances. Leaving fpr matrix as None')
            self.fpr = None
            return

        fpr = np.zeros((len(self.iou),len(self.rc)))
        for i in range(len(self.iou)):
            for j,s in enumerate(self.scores[i]): #j -> recall level, s -> corresponding score
                ng = 0 #number of negative images
                fp = 0 #number of false positives images
                for el in self.mycoco.evalImgs:
                    if el is None:#no predictions, no gts
                        ng=ng+1
                    elif len(el['gtIds'])==0:# some predictions and no gts
                        ng=ng+1
                        if (np.array(el['dtScores']) >s).sum() > 0: #if at least one score over threshold for recall level
                            fp=fp+1 #count as FP
                    else:
                        continue
                fpr[i,j] = fp/ng
        self.fpr = fpr 

    def _calculate_fpr(self):
        print('Using alternate calculation for fpr at instance score threshold of {}'.format(self.prob_thresh))
        ng = 0 #number of negative images
        fp = 0 #number of false positives images
        for el in self.mycoco.evalImgs:
            if el is None:#no predictions, no gts
                ng=ng+1
            elif len(el['gtIds'])==0:# some predictions and no gts
                ng=ng+1
                if (np.array(el['dtScores']) >self.prob_thresh).sum() > 0: #if at least one score over threshold for recall level
                    fp=fp+1 #count as FP
            else: #gt has instances
                continue
        return fp/(ng+1e-5)

    def _find_iou_rc_inds(self):
        try:
            iou_idx = np.argwhere(self.iou==self.iou_thresh)[0][0] #first instance of
        except IndexError:
            print('iou threshold {} not found in mycoco.params.iouThrs {}'.format(self.iou_thresh,self.mycoco.params.iouThrs))
            exit(1)
        #test above for out of bounds
        inds = np.argwhere(self.scores[iou_idx]>=self.prob_thresh)
        if len(inds)>0:
            rc_idx = inds[-1][0] #get recall index corresponding to prob_thresh
        else:
            rc_idx = 0
        return iou_idx,rc_idx

    def get_fpr(self):
        
        if self.fpr is None:
            self._calculate_fpr_matrix()

        if self.fpr is not None: 
            iou_idx,rc_idx = self._find_iou_rc_inds()
            fpr = self.fpr[iou_idx,rc_idx]
        elif len(self.mycoco.cocoGt.anns)==0:
            fpr = self._calculate_fpr()
        else:
            fpr=-1
        return fpr
    
    def summarize_scalars(self): #for pretty printing 
        p,r = self.get_precision_recall()
        fpr = self.get_fpr()
        dd = dict(dataset = self.dataset_name, precision = float(p),recall=float(r),fpr=float(fpr),iou=self.iou_thresh,probability=self.prob_thresh)
        return dd

from sklearn.metrics import precision_recall_curve,average_precision_score
class CreatePlotsRPD():
    def __init__(self,dfimg):
        self.dfimg = dfimg
        self.dfpts = self.dfimg.groupby(['ptid','eye'])[['gt_instances','gt_pxs','gt_xpxs','dt_instances','dt_pxs','dt_xpxs']].sum()
        
    @classmethod
    def initfromcoco(cls,mycoco,prob_thresh):
        df = pd.DataFrame(index = mycoco.cocoGt.imgs.keys(), columns=['gt_instances','gt_pxs','gt_xpxs','dt_instances','dt_pxs','dt_xpxs'],dtype=np.uint64)

        for key,val in mycoco.cocoGt.imgs.items():
            imgid = val['id']
            #Gt instances
            annIdsGt = mycoco.cocoGt.getAnnIds([imgid])
            annsGt = mycoco.cocoGt.loadAnns(annIdsGt)
            instGt = [mycoco.cocoGt.annToMask(ann).sum() for ann in annsGt]
            xprojGt = [(mycoco.cocoGt.annToMask(ann).sum(axis=0)>0).astype('uint8').sum() for ann in annsGt] 
            #Dt instances
            annIdsDt = mycoco.cocoDt.getAnnIds([imgid])
            annsDt = mycoco.cocoDt.loadAnns(annIdsDt)
            annsDt = [ann for ann in annsDt if ann['score']>prob_thresh]
            instDt = [mycoco.cocoDt.annToMask(ann).sum() for ann in annsDt]
            xprojDt = [(mycoco.cocoDt.annToMask(ann).sum(axis=0)>0).astype('uint8').sum() for ann in annsDt]
                
            dat = [len(instGt),np.array(instGt).sum(),np.array(xprojGt).sum(),len(instDt),np.array(instDt).sum(),np.array(xprojDt).sum()]
            df.loc[key] = dat
            
        newdf = pd.DataFrame([idx.strip('.png').split('_') for idx in df.index],columns=['ptid','eye','scan'],index = df.index)
        df = df.merge(newdf,how='inner',left_index=True,right_index=True)
        return cls(df)
    
    @classmethod
    def initfromcsv(cls,fname):
        df = pd.read_csv(fname)
        return cls(df)
    
            
    def get_max_limits(self,df):
        max_inst=np.max([df.gt_instances.max(),df.dt_instances.max()])
        max_xpxs = np.max([df.gt_xpxs.max(),df.dt_xpxs.max()])
        max_pxs = np.max([df.gt_pxs.max(),df.dt_pxs.max()])
#         print('Max instances:',max_inst)
#         print('Max xpxs:',max_xpxs)
#         print('Max pxs:',max_pxs)
        return max_inst,max_xpxs,max_pxs

    def eye_level_prc(self,df,gt_thresh=5,ax=None):

        prc = precision_recall_curve(df.gt_instances>=gt_thresh,df.dt_instances)
        if ax==None:
            fig,ax = plt.subplots(1,1)
        ax.plot(prc[1],prc[0])
        ax.set_xlabel('RPD Eye Recall')
        ax.set_ylabel('RPD Eye Precision')
        fig2,ax2 = plt.subplots(1,1)
        ax2.plot(prc[1][:-1],prc[2])
        ax2.set_ylabel('RPD Instance Threshold')
        ax2.set_xlabel('RPD Eye Recall')

        ap = average_precision_score(df.gt_instances>=gt_thresh,df.dt_instances)
        return ap,prc

    def plot_img_level_instance_thresholding(self,df,inst):

        rc = np.zeros((len(inst),))
        pr = np.zeros((len(inst),))
        fpr = np.zeros((len(inst),))

        fig, ax = plt.subplots(1,3,figsize = [15,5])
        for i,dt_thresh in enumerate(inst):
            gt = df.gt_instances>dt_thresh
            dt = df.dt_instances>dt_thresh
            rc[i] = (gt&dt).sum()/gt.sum()
            pr[i] = (gt&dt).sum()/dt.sum()
            fpr[i] = ((~gt)&(dt)).sum()/((~gt).sum())

        ax[1].plot(inst,pr)
        ax[1].set_ylim(0.45,1.01)
        ax[1].set_xlabel('instance threshold')
        ax[1].set_ylabel('Precision')


        ax[0].plot(inst,rc)
        ax[0].set_ylim(0.45,1.01)
        ax[0].set_ylabel('Recall')
        ax[0].set_xlabel('instance threshold')


        ax[2].plot(inst,fpr)
        ax[2].set_ylim(0,0.80)
        ax[2].set_xlabel('instance threshold')
        ax[2].set_ylabel('FPR')

        plt.tight_layout()
        return pr,rc,fpr

    def plot_img_level_instance_thresholding2(self,df,inst,gt_thresh):

        rc = np.zeros((len(inst),))
        pr = np.zeros((len(inst),))
        fpr = np.zeros((len(inst),))

        fig, ax = plt.subplots(1,3,figsize = [15,5])
        for i,dt_thresh in enumerate(inst):
            gt = df.gt_instances>=gt_thresh
            dt = df.dt_instances>=dt_thresh
            rc[i] = (gt&dt).sum()/gt.sum()
            pr[i] = (gt&dt).sum()/dt.sum()
            fpr[i] = ((~gt)&(dt)).sum()/((~gt).sum())

        ax[1].plot(inst,pr)
        ax[1].set_ylim(0.45,1.01)
        ax[1].set_xlabel('instance threshold')
        ax[1].set_ylabel('Precision')


        ax[0].plot(inst,rc)
        ax[0].set_ylim(0.45,1.01)
        ax[0].set_ylabel('Recall')
        ax[0].set_xlabel('instance threshold')


        ax[2].plot(inst,fpr)
        ax[2].set_ylim(0,0.80)
        ax[2].set_xlabel('instance threshold')
        ax[2].set_ylabel('FPR')

        plt.tight_layout()
        return pr,rc,fpr

    def gt_vs_dt_instances(self,ax=None):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        
        if ax==None:
            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(111)

        y = df[idx].groupby('gt_instances')['dt_instances'].mean()
        yerr = df[idx].groupby('gt_instances')['dt_instances'].std()
        ax.errorbar(y.index,y.values,yerr.values,fmt='*')
        plt.plot([0,max_inst],[0,max_inst],alpha=.5)
        plt.xlim(0,max_inst+1)
        plt.ylim(0,max_inst+1)
        ax.set_aspect(1)
        plt.xlabel('gt_instances')
        plt.ylabel('dt_instances')
        plt.tight_layout()
        return ax
        
    def gt_vs_dt_xpxs(self):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        dfsub = df[idx]
        
        fig1 = plt.figure(figsize = [10,10],dpi=100)
        ax = fig1.add_subplot(111)
        sc = ax.scatter(dfsub['gt_xpxs'],dfsub['dt_xpxs'],c =dfsub['gt_instances'] ,cmap='viridis')
        ax.set_aspect(1)
        #ax = dfsub.plot(kind = 'scatter',x=,y=,c='gt_instances')
        plt.plot([0,max_xpxs],[0,max_xpxs],alpha=.5)
        plt.xlim(0,max_xpxs)
        plt.ylim(0,max_xpxs)
        plt.xlabel('gt_xpxs')
        plt.ylabel('dt_xpxs')
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('gt_instances')
        plt.tight_layout()
        
        fig2 = plt.figure(figsize = [10,10],dpi=100)
        ax = fig2.add_subplot(111)
        sc = ax.scatter(dfsub['gt_xpxs'],dfsub['gt_xpxs']-dfsub['dt_xpxs'],c =dfsub['gt_instances'] ,cmap='viridis')
        #ax = dfsub.plot(kind = 'scatter',x=,y=,c='gt_instances')
        plt.plot([0,max_xpxs],[0,0],alpha=.5)
        plt.xlabel('gt_xpxs')
        plt.ylabel('gt_xpxs-dt_xpxs')
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('gt_instances')
        plt.tight_layout()
        
        fig3 = plt.figure(dpi=100)
        plt.hist(dfsub['gt_xpxs']-dfsub['dt_xpxs'])
        plt.xlabel('gt_xpxs - dt_xpxs')
        plt.ylabel('B-scans')
        
        return fig1,fig2,fig3
    
    def gt_vs_dt_xpxs_mu(self):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        dfsub = df[idx]
        
        from scipy import stats
        mu_dt,bins,bnum = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['dt_xpxs'],statistic = 'mean',bins=10)
        std_dt,_,_ = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['dt_xpxs'],statistic = 'std',bins = bins)
        mu_gt,_,_ = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['gt_xpxs'],statistic='mean',bins=bins)
        std_gt,_,_ = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['gt_xpxs'],statistic = 'std',bins = bins)
        fig = plt.figure(dpi=100)
        plt.errorbar(mu_gt,mu_dt,yerr = std_dt,xerr=std_gt,fmt='*')
        plt.xlabel('gt_xpxs')
        plt.ylabel('dt_xpxs')
        plt.plot([0,max_xpxs],[0,max_xpxs],alpha=.5)
        plt.xlim(0,max_xpxs)
        plt.ylim(0,max_xpxs)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        return fig

    def gt_dt_FP_FN_count(self):
        df = self.dfimg
        fig,ax =plt.subplots(1,2,figsize=[10,5])

        idx = (df.gt_instances==0)&(df.dt_instances>0)
        ax[0].hist(df[idx]['dt_instances'],bins = range(1,10))
        ax[0].set_xlabel('dt instances')
        ax[0].set_ylabel('B-scans')
        ax[0].set_title('FP dt instance count per B-scan')

        idx = (df.gt_instances>0)&(df.dt_instances==0)
        ax[1].hist(df[idx]['gt_instances'],bins = range(1,10))
        ax[1].set_xlabel('gt instances')
        ax[1].set_ylabel('B-scans')
        ax[1].set_title('FN gt instance count per B-scan')

        plt.tight_layout()
        return fig
    
    def avg_inst_size(self):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        dfsub = df[idx]
        
        fig = plt.figure(figsize=[10,5])
        plt.subplot(121)
        bins = np.arange(0,120,10)
        ax = (dfsub.gt_xpxs/dfsub.gt_instances).hist(bins = bins,alpha=.5,label='gt')
        ax = (dfsub.dt_xpxs/dfsub.dt_instances).hist(bins=bins,alpha=.5,label='dt')
        ax.set_xlabel('xpxs')
        ax.set_ylabel('B-scans')
        ax.set_title('Average size of instance')
        ax.legend()

        plt.subplot(122)
        bins = np.arange(0,600,40)
        ax = (dfsub.gt_pxs/dfsub.gt_instances).hist(bins=bins,alpha=.5,label='gt')
        ax = (dfsub.dt_pxs/dfsub.dt_instances).hist(bins=bins,alpha=.5,label='dt')
        ax.set_xlabel('pxs')
        ax.set_ylabel('B-scans')
        ax.set_title('Average size of instance')
        ax.legend()

        plt.tight_layout()
        return fig
    
def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)

        myeval = EvaluateClass(
            dataset_name, cfg.OUTPUT_DIR,iou_thresh = .1,prob_thresh=0.5)
        results_i = inference_on_dataset(model, data_loader, myeval)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {}:".format(dataset_name))
            logger.info('Precision: {} \t Recall {}'.format(results_i[0],results_i[1]))
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    # data_loader = build_detection_train_loader(cfg,
    #    mapper=DatasetMapper(cfg, is_train=True, augmentations=[
    #         T.RandomBrightness(.9, 1.1),
    #         T.RandomFlip(prob=0.5),
    #         T.RandomRotation([-10,10]),
    #         T.RandomContrast(.8,1.2)
    #    ]))

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        results = do_test(cfg, model)
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                results = do_test(cfg, model)
                #if comm.is_main_process()?
                for dataset_name in cfg.DATASETS.TEST:
                    if comm.is_main_process():
                        storage.put_scalar(dataset_name+"/pr",results[dataset_name][0],smoothing_hint=False)
                        storage.put_scalar(dataset_name+"/rc",results[dataset_name][1],smoothing_hint=False)
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    # DatasetCatalog.register("rpd_train", grab_train)
    # DatasetCatalog.register("rpd_valid", grab_valid)
    # MetadataCatalog.get("rpd_valid").thing_classes = ["rpd"]

    cfg = setup(args)
    for name in cfg.DATASETS.TRAIN:
        DatasetCatalog.register(name, grab_dataset(name))
        MetadataCatalog.get(name).thing_classes = ["rpd"]
    for name in cfg.DATASETS.TEST:
        DatasetCatalog.register(name,grab_dataset(name))
        MetadataCatalog.get(name).thing_classes = ["rpd"]

    #check split right before running model
    dfcheck = checkSplits(cfg.DATASETS.TRAIN+cfg.DATASETS.TEST)
    print(dfcheck)
    if (dfcheck.values.sum() - np.diag(dfcheck.values).sum())>0:
        raise ValueError('There are overlapping folds!')

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

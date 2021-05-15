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
import torch
from torch.nn.parallel import DistributedDataParallel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval,Params

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
# @contextlib.contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = io.BytesIO()
#     yield
#     sys.stdout = save_stdout

def grab_train():
    return pickle.load( open( "datasets/train_refined.pk", "rb" ) )
def grab_valid():
    return pickle.load( open( "datasets/valid_refined.pk", "rb" ) )

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import cv2
from PIL import Image
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ybpres.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages


class OutputVis():
    
    def __init__(self,dataset_name,cfg,prob_thresh):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.prob_thresh = prob_thresh
        self.data = DatasetCatalog.get(dataset_name)
        self.predictor = DefaultPredictor(cfg)
        

    def get_image(self,ImgId):
        gt_data = next(item for item in self.data if (item['image_id'] == ImgId))
        
        dat = gt_data #gt
        im = cv2.imread(dat['file_name']) #input to model
        bboxes = [ddict['bbox'] for ddict in dat['annotations']]
        BBoxes = detectron2.structures.Boxes(bboxes)
        BBoxes = detectron2.structures.BoxMode.convert(BBoxes.tensor,from_mode=1,to_mode=0) #1= XYXY, 2 = XYWH
        segs = [ddict['segmentation'] for ddict in dat['annotations']]
        v = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=3.0)
        result_image = v.overlay_instances(boxes=BBoxes,masks=segs).get_image()
        img = Image.fromarray(result_image)
        
        outputs = self.predictor(im)["instances"].to("cpu")
        outputs = outputs[outputs.scores>self.prob_thresh]
        v2 = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=3.0)
        v2._default_font_size = 14
        result_model = v2.draw_instance_predictions(outputs).get_image()
        img_model = Image.fromarray(result_model)
        
        return img, img_model
    
    def output_to_pdf(self,ImgIds,outname,dfimg=None):
        
        def height_crop_range(im,height_target=256):
            yhist = im.sum(axis=1) #integrate over width of image
            mu = np.average(np.arange(yhist.shape[0]),weights = yhist)
            h1 = int(np.floor(mu-height_target/2))
            h2 = int(np.ceil(mu+height_target/2))
            return range(h1,h2)
        
        gtstr = ''
        dtstr = ''
        
        if dfimg is not None:
            gtcols = dfimg.columns[['gt_' in col for col in dfimg.columns]]
            dtcols = dfimg.columns[['dt_' in col for col in dfimg.columns]]
            
        with PdfPages(outname) as pdf:
            for imgid in tqdm(ImgIds):
                img, img_model = self.get_image(imgid)
                #pdb.set_trace()
                crop_range = height_crop_range(np.array(img.convert('L')),height_target=256*3)
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
            
#             imgpage = Image.fromarray(np.concatenate((img,img_model),axis=0))
#             if os.path.exists(outname):
#                 imgpage.save(outname,'PDF',resolution=300,append=True)
#             else:
#                 imgpage.save(outname,'PDF',resolution=300)
            
                

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
        self.dfimg=None
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
        self.cocoGt = COCO(os.path.join(self._output_dir,self.dataset_name +'_coco_format.json'))
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
        self.scores = self.mycoco.eval['scores'][:,:,0,0,0]
        p,r = self.get_precision_recall()
        return p,r

    def plot_PRcurve(self):
        for i in range(len(self.iou)):
            plt.plot(self.rc,self.pr[i],label = '{:.2}'.format(self.iou[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('')
        plt.legend(title='IoU')
        
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
        fpr = np.zeros((len(self.iou),len(self.rc)))
        for i in range(len(self.iou)):
            for j,s in enumerate(self.scores[i]):
                ng = 0
                fp = 0
                for el in self.mycoco.evalImgs:
                    if el is None:#no predictions
                        ng=ng+1
                    elif len(el['gtIds'])==0:# some predictions
                        ng=ng+1
                        if (np.array(el['dtScores']) >s).sum() > 0:
                            fp=fp+1
                    else:
                        continue
                fpr[i,j] = fp/ng
        self.fpr = fpr 
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
        iou_idx,rc_idx = self._find_iou_rc_inds()
        if self.fpr is None:
            self._calculate_fpr_matrix()
        return self.fpr[iou_idx,rc_idx]
    
    def summarize_scalars(self): #for pretty printing 
        p,r = self.get_precision_recall()
        fpr = self.get_fpr()
        dd = dict(dataset = self.dataset_name, precision = p,recall=r,fpr=fpr,iou=self.iou_thresh,probability=self.prob_thresh)
        return dd

    def build_dfimg(self):
        df = pd.DataFrame(index = self.cocoGt.imgs.keys(), columns=['gt_instances','gt_pxs','gt_xpxs','dt_instances','dt_pxs','dt_xpxs'],dtype=np.uint64)

        for key,val in self.cocoGt.imgs.items():
            imgid = val['id']
            #Gt instances
            annIdsGt = self.mycoco.cocoGt.getAnnIds([imgid])
            annsGt = self.mycoco.cocoGt.loadAnns(annIdsGt)
            instGt = [self.mycoco.cocoGt.annToMask(ann).sum() for ann in annsGt]
            xprojGt = [(self.mycoco.cocoGt.annToMask(ann).sum(axis=0)>0).astype('uint8').sum() for ann in annsGt] 
            #Dt instances
            annIdsDt = self.mycoco.cocoDt.getAnnIds([imgid])
            annsDt = self.mycoco.cocoDt.loadAnns(annIdsDt)
            annsDt = [ann for ann in annsDt if ann['score']>self.prob_thresh]
            instDt = [self.mycoco.cocoDt.annToMask(ann).sum() for ann in annsDt]
            xprojDt = [(self.mycoco.cocoDt.annToMask(ann).sum(axis=0)>0).astype('uint8').sum() for ann in annsDt]
                
            dat = [len(instGt),np.array(instGt).sum(),np.array(xprojGt).sum(),len(instDt),np.array(instDt).sum(),np.array(xprojDt).sum()]
            df.loc[key] = dat
            self.dfimg = df



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
    data_loader = build_detection_train_loader(cfg,
       mapper=DatasetMapper(cfg, is_train=True, augmentations=[
            T.RandomBrightness(.9, 1.1),
            T.RandomFlip(prob=0.5),
            T.RandomRotation([-10,10]),
            T.RandomContrast(.8,1.2)
       ]))

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
    DatasetCatalog.register("rpd_train", grab_train)
    DatasetCatalog.register("rpd_valid", grab_valid)
    MetadataCatalog.get("rpd_valid").thing_classes = ["rpd"]

    cfg = setup(args)

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

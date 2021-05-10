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
from pycocotools.cocoeval import COCOeval,Params

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
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
import pdb
import warnings

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

class EvaluateClass(COCOEvaluator):
    
    def __init__(self,dataset_name, output_dir,prob_thresh=0.5,iou_thresh = 0.1):
        super().__init__(dataset_name,tasks={'bbox'},output_dir = output_dir)
        self.mycoco=None
        self.prob_thresh = prob_thresh #instance probabilty threshold for scalars
        self.iou_thresh = iou_thresh    
    def reset(self): 
        super().reset()
        self.mycoco=None
    def process(self, inputs, outputs):
        super().process(inputs,outputs)
    def evaluate(self):
        #with nostdout(): #suppress the coco summarize statment (the one with APs)
        _ = super().evaluate() #this call populates coco_instances_results.json
        comm.synchronize()
        if not comm.is_main_process():
            return ()
        self.cocoDt = self._coco_api.loadRes(os.path.join(self._output_dir,'coco_instances_results.json')) #load detector results
        self.mycoco = COCOeval(self._coco_api,self.cocoDt,iouType ='bbox')
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

        #pdb.set_trace()
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
        precision = self.pr[iou_idx,rc_idx]
        recall = self.rc[rc_idx]
        return precision,recall


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
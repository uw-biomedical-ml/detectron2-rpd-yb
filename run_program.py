import datasets.data as data
import plain_train_net as net
import pickle
import matplotlib.pyplot as plt
import numpy as np
from detectron2.config import get_cfg
from plain_train_net import grab_dataset
import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.engine import launch
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from Ensembler import Ensembler
from plain_train_net import EvaluateClass,CreatePlotsRPD
from plain_train_net import OutputVis
from PIL import Image,ImageOps,ImageFilter,ImageSequence
from detectron2.utils.visualizer import Visualizer
import json
import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

has_annotations = False
dataset_name = "Test Set"
dpi= 120
dataset_table = None
cfg = None


def process_input(has_annotations=False): # Processes input .vol files and creates the pk file.
    data.rpath='/data/ssong/rpd_data' #root path for files, should be changed to user input
    data.dirtoextract = data.rpath +'/Test' #extracted from, should be changed to user input
    data.filedir = data.rpath +'/Test_extracted' #split from, should be changed to user input
    data.extractFiles(masks_exist = has_annotations)
    df = data.createDf()
    df.assign(fold = dataset_name) #temporary
    df_p = data.process_masks(df, mode = 'binary', binary_classes=2)
    stored_data = data.rpd_data(df_p, grp = dataset_name, data_has_ann = has_annotations) #temp handling of grp
    pickle.dump(stored_data, open(os.path.join(data.script_dir,f"{dataset_name}_refined.pk"), "wb")) #temp handling of grp

def configure_model():
    global cfg
    cfg = get_cfg()
    cfg.merge_from_file('configs/working.yaml')
    for name in [dataset_name]:
        try:
            DatasetCatalog.register(name, grab_dataset(name))
        except:
            print('Already registered.')
        MetadataCatalog.get(name).thing_classes = ["rpd"]

def run_prediction(cfg,dataset_name):
    model = build_model(cfg)  # returns a torch.nn.Module
    myloader = build_detection_test_loader(cfg,dataset_name) 
    myeval = COCOEvaluator(dataset_name,tasks={'bbox','segm'},output_dir ="output_"+ dataset_name) #produces _coco_format.json when initialized
    for mdl in ("fold1", "fold2", "fold3", "fold4","fold5"):
        model_weights_path = "/data/amd-data/cera-rpd/detectron2-rpd/output_valid_"+ mdl +"/model_final.pth"
        DetectionCheckpointer(model).load(model_weights_path);  # load a file, usually from cfg.MODEL.WEIGHTS
        model.eval(); #set model in evaluation mode
        myeval.reset()
        output_dir = "output_"+ dataset_name + "/"+mdl
        myeval._output_dir = output_dir
        #print("running inference on ", mdl)
        results_i = inference_on_dataset(model, myloader, myeval) #produces coco_instance_results.json when myeval.evaluate is called
    print("Done with predictions!")
    return

def run_ensemble():
    ens = Ensembler('output_'+dataset_name,dataset_name,["fold1", "fold2", "fold3", "fold4","fold5"],.2)
    ens.mean_score_nms()
    ens.save_coco_instances()

def evaluate_ensemble():
    myeval = EvaluateClass(dataset_name, "output_"+ dataset_name, iou_thresh = .2, prob_thresh=0.5,evalsuper=False)
    myeval.evaluate()
    #print(myeval.summarize_scalars())
    with open(os.path.join("output_"+ dataset_name,'scalar_dict.json'),"w") as outfile:
        json.dump(obj=myeval.summarize_scalars(),fp=outfile)
    global dataset_table 
    dataset_table = CreatePlotsRPD.initfromcoco(myeval.mycoco,myeval.prob_thresh)

def create_binary_masks_tif():
    pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
    dfimg_dummy = dataset_table.dfimg.sort_index()
    df_unique = dfimg_dummy.iloc[::49, :]
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=has_annotations)
    for scan in range(len(df_unique.index)):
        ImgIds = dfimg_dummy.head(49).index.values
        vis.output_masks_to_tiff(ImgIds, dfimg_dummy.loc[ImgIds[0],"ptid"], dfimg_dummy.loc[ImgIds[0], "eye"])
        dfimg_dummy = dfimg_dummy.iloc[49:]
        if dfimg_dummy.empty:
            #print('DataFrame is empty!')
            break

def create_binary_masks_overlay_tif():
    pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
    dfimg_dummy = dataset_table.dfimg.sort_index()
    df_unique = dfimg_dummy.iloc[::49, :]
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=has_annotations)
    for scan in range(len(df_unique.index)):
        ImgIds = dfimg_dummy.head(49).index.values
        vis.output_overlay_masks_to_tiff(ImgIds, dfimg_dummy.loc[ImgIds[0],"ptid"], dfimg_dummy.loc[ImgIds[0], "eye"])
        dfimg_dummy = dfimg_dummy.iloc[49:]
        if dfimg_dummy.empty:
            #print('DataFrame is empty!')
            break

def create_instance_masks_overlay_tif():
    pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
    dfimg_dummy = dataset_table.dfimg.sort_index()
    df_unique = dfimg_dummy.iloc[::49, :]
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=has_annotations)
    for scan in range(len(df_unique.index)):
        ImgIds = dfimg_dummy.head(49).index.values
        vis.output_instances_masks_to_tiff(ImgIds, dfimg_dummy.loc[ImgIds[0],"ptid"], dfimg_dummy.loc[ImgIds[0], "eye"])
        dfimg_dummy = dfimg_dummy.iloc[49:]
        if dfimg_dummy.empty:
            #print('DataFrame is empty!')
            break
def main():
    global has_annotations
    has_annotations = False
    print("Has annotations: ", has_annotations)
    print("Processing input...")
    process_input(has_annotations = has_annotations)
    print("Configuring model...")
    configure_model()
    print("Running inference...")
    run_prediction(cfg,dataset_name)
    print("Running ensemble...")
    run_ensemble()
    print("Evaluating ensemble...")
    evaluate_ensemble()
    print("Creating binary masks tif (no overlay)...")
    create_binary_masks_tif()
    print("Creating binary masks tif (with overlay)...")
    create_binary_masks_overlay_tif()
    print("Creating instances masks tif (with overlay)...")
    create_instance_masks_overlay_tif()
    print("Done!")

if __name__ == "__main__":
    main()
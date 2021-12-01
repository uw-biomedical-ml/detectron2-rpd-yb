import datasets.data as data
import plain_train_net as net
import pickle
import matplotlib.pyplot as plt
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
from table_styles import styles

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

has_annotations = False
dataset_name = "Fold 1"
dpi= 120
dataset_table = None
cfg = None
myeval = None
ens = None


def process_input(has_annotations=False): # Processes input .vol files and creates the pk file.
    data.rpath='C:/Users/scott/Desktop/Lee Lab Research' #root path for files, should be changed to user input
    # data.dirtoextract = data.rpath +'/Test' #extracted from, should be changed to user input
    data.filedir = data.rpath +'/data/DRUSEN' #split from, should be changed to user input
    # data.extractFiles(masks_exist = has_annotations)
    df_p = data.createDfTest().assign(fold = dataset_name) #temporary
    # df_p = data.process_masks(df, mode = 'binary', binary_classes=2)
    stored_data = data.rpd_data(df_p, grp = dataset_name, data_has_ann = has_annotations) #temp handling of grp
    pickle.dump(stored_data, open(os.path.join(data.script_dir,f"{dataset_name}_refined.pk"), "wb")) #temp handling of grp

def configure_model():
    global cfg
    cfg = get_cfg()
    cfg.merge_from_file('configs/working.yaml')

def register_dataset():
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
        # model_weights_path = "/data/amd-data/cera-rpd/detectron2-rpd/output_valid_"+ mdl +"/model_final.pth"
        model_weights_path = 'C:/Users/scott/Desktop/Lee Lab Research' + "/models/"+ mdl +"/model_final.pth"
        DetectionCheckpointer(model).load(model_weights_path);  # load a file, usually from cfg.MODEL.WEIGHTS
        model.eval() #set model in evaluation mode
        myeval.reset()
        output_dir = "output_"+ dataset_name + "/"+mdl
        myeval._output_dir = output_dir
        print("Running inference with model ", mdl)
        results_i = inference_on_dataset(model, myloader, myeval) #produces coco_instance_results.json when myeval.evaluate is called
    print("Done with predictions!")

def run_ensemble():
    global ens
    ens = Ensembler('output_'+dataset_name,dataset_name,["fold1", "fold2", "fold3", "fold4","fold5"],.2)
    ens.mean_score_nms()
    ens.save_coco_instances()

def evaluate_dataset():
    global myeval
    myeval = EvaluateClass(dataset_name, "output_"+ dataset_name, iou_thresh = .2, prob_thresh=0.5,evalsuper=False)
    myeval.evaluate()
    with open(os.path.join("output_"+ dataset_name,'scalar_dict.json'),"w") as outfile:
        json.dump(obj=myeval.summarize_scalars(),fp=outfile)

def create_table():
    if (myeval == None):
        evaluate_dataset()
    global dataset_table
    dataset_table = CreatePlotsRPD.initfromcoco(myeval.mycoco,myeval.prob_thresh)

def create_binary_masks_tif():
    pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
    dfimg_dummy = dataset_table.dfimg.sort_index()
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=True)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD']
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS']
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            vis.output_masks_to_tiff(df_pt_OD_ids, df_unique[scan], 'OD')
        if (len(df_pt_OS.index) > 0):
            vis.output_masks_to_tiff(df_pt_OS_ids, df_unique[scan], 'OS')

def create_binary_masks_overlay_tif():
    pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
    dfimg_dummy = dataset_table.dfimg.sort_index()
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=True)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD']
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS']
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            vis.output_overlay_masks_to_tiff(df_pt_OD_ids, df_unique[scan], 'OD')
        if (len(df_pt_OS.index) > 0):
            vis.output_overlay_masks_to_tiff(df_pt_OS_ids, df_unique[scan], 'OS')

def create_instance_masks_overlay_tif():
    pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
    dfimg_dummy = dataset_table.dfimg.sort_index()
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=True)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD']
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS']
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            vis.output_instances_masks_to_tiff(df_pt_OD_ids, df_unique[scan], 'OD')
        if (len(df_pt_OS.index) > 0):
            vis.output_instances_masks_to_tiff(df_pt_OS_ids, df_unique[scan], 'OS')

def create_tif_output(mode = None):
    pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
    dfimg_dummy = dataset_table.dfimg.sort_index()
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=True)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD']
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS']
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            if (mode == 'bm'):
                vis.output_masks_to_tiff(df_pt_OD_ids, df_unique[scan], 'OD')
            elif (mode == 'bm-o'):
                vis.output_overlay_masks_to_tiff(df_pt_OD_ids, df_unique[scan], 'OD')
            elif (mode == 'im'):
                vis.output_instances_masks_to_tiff(df_pt_OD_ids, df_unique[scan], 'OD')
            else:
                print("No output mode selected!")
        if (len(df_pt_OS.index) > 0):
            if (mode == 'bm'):
                vis.output_masks_to_tiff(df_pt_OS_ids, df_unique[scan], 'OS')
            elif (mode == 'bm-o'):
                vis.output_overlay_masks_to_tiff(df_pt_OS_ids, df_unique[scan], 'OS')
            elif (mode == 'im'):
                vis.output_instances_masks_to_tiff(df_pt_OS_ids, df_unique[scan], 'OS')
            else:
                print("No output mode selected!")
def create_dfpts():
    if (dataset_table == None):
        create_table()
    dfpts = dataset_table.dfpts.sort_values(by=['dt_instances'],ascending=False)
    html_str = dfpts.style.format('{:.0f}').set_table_styles(styles).render()
    html_file = open(os.path.join('output_'+ dataset_name + '/dfpts_'+dataset_name+'.html'),'w')
    html_file.write(html_str)
    html_file.close()

def create_dfimg():
    if (dataset_table == None):
        create_table()
    dfimg = dataset_table.dfimg.sort_index()
    html_str = dfimg.style.set_table_styles(styles).render()
    html_file = open(os.path.join('output_'+ dataset_name + '/dfimg_'+dataset_name+'.html'),'w')
    html_file.write(html_str)
    html_file.close()

def main():
    global has_annotations
    has_annotations = False
    print("Has annotations: ", has_annotations)
    print("Processing input...")
    process_input(has_annotations = has_annotations)
    print("Configuring model...")
    configure_model()
    print("Registering dataset...")
    register_dataset()
    print("Running inference...")
    run_prediction(cfg,dataset_name)
    print("Running ensemble...")
    run_ensemble()
    print("Evaluating dataset...")
    evaluate_dataset()
    print("Creating dataset table...")
    create_table()
    print("Creating binary masks tif (no overlay)...")
    create_tif_output(mode = 'bm')
    print("Creating binary masks tif (with overlay)...")
    create_tif_output(mode = 'bm-o')
    print("Creating instances masks tif (with overlay)...")
    create_tif_output(mode = 'im')
    # create_dfpts()
    # create_dfimg()
    print("Done!")

if __name__ == "__main__":
    main()
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
import argparse
import urllib.request
import zipfile

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_name = None
dpi= 120
dataset_table = None
cfg = None
myeval = None
ens = None
output_path = None


def process_input(): # Processes input .vol files and creates the pk file.
    data.extractFiles(name = dataset_name)
    stored_data = data.rpd_data(name = dataset_name)
    pickle.dump(stored_data, open(os.path.join(data.script_dir,f"{dataset_name}_refined.pk"), "wb"))

def configure_model():
    global cfg
    cfg = get_cfg()
    moddir = os.path.dirname(os.path.realpath(__file__))
    name = 'working.yaml'
    cfg_path = os.path.join(moddir, name)
    cfg.merge_from_file(cfg_path)

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
    myeval = COCOEvaluator(dataset_name,tasks={'bbox','segm'},output_dir =output_path) #produces _coco_format.json when initialized
    for mdl in ("fold1", "fold2", "fold3", "fold4","fold5"):
        extract_directory = 'models_t'
        if not os.path.isdir(extract_directory):
            os.mkdir(extract_directory)
            url = 'https://s3.us-west-2.amazonaws.com/comp.ophthalmology.uw.edu/models.zip'
            path_to_zip_file, headers = urllib.request.urlretrieve(url)
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_directory)
        file_name = mdl + "_model_final.pth"
        model_weights_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), extract_directory, file_name)
        print(model_weights_path)
        DetectionCheckpointer(model).load(model_weights_path) # load a file, usually from cfg.MODEL.WEIGHTS
        model.eval() #set model in evaluation mode
        myeval.reset()
        output_dir = os.path.join(output_path, mdl)
        myeval._output_dir = output_dir
        print("Running inference with model ", mdl)
        results_i = inference_on_dataset(model, myloader, myeval) #produces coco_instance_results.json when myeval.evaluate is called
    print("Done with predictions!")

def run_ensemble():
    global ens
    ens = Ensembler(output_path,dataset_name,["fold1", "fold2", "fold3", "fold4","fold5"],.2)
    ens.mean_score_nms()
    ens.save_coco_instances()

def evaluate_dataset():
    global myeval
    myeval = EvaluateClass(dataset_name, output_path, iou_thresh = .2, prob_thresh=0.5,evalsuper=False)
    myeval.evaluate()
    with open(os.path.join(output_path,'scalar_dict.json'),"w") as outfile:
        json.dump(obj=myeval.summarize_scalars(),fp=outfile)

def create_table():
    if (myeval == None):
        evaluate_dataset()
    global dataset_table
    dataset_table = CreatePlotsRPD.initfromcoco(myeval.mycoco,myeval.prob_thresh)
    dataset_table.dfimg.sort_index(inplace=True)
    #dataset_table.dfimg['scan'] = dataset_table.dfimg['scan'].astype('int') #depends on what we want scan field to be

def create_binary_masks_tif():
    pred_file = os.path.join(output_path, 'coco_instances_results.json')
    dfimg_dummy = dataset_table.dfimg
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD'].sort_values('scan', kind = 'mergesort')
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS'].sort_values('scan', kind = 'mergesort')
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            vis.output_masks_to_tiff(output_path, df_pt_OD_ids, df_unique[scan], 'OD')
        if (len(df_pt_OS.index) > 0):
            vis.output_masks_to_tiff(output_path, df_pt_OS_ids, df_unique[scan], 'OS')

def create_binary_masks_overlay_tif():
    pred_file = os.path.join(output_path, 'coco_instances_results.json')
    dfimg_dummy = dataset_table.dfimg
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD'].sort_values('scan', kind = 'mergesort')
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS'].sort_values('scan', kind = 'mergesort')
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            vis.output_overlay_masks_to_tiff(output_path, df_pt_OD_ids, df_unique[scan], 'OD')
        if (len(df_pt_OS.index) > 0):
            vis.output_overlay_masks_to_tiff(output_path, df_pt_OS_ids, df_unique[scan], 'OS')

def create_instance_masks_overlay_tif():
    pred_file = os.path.join(output_path, 'coco_instances_results.json')
    dfimg_dummy = dataset_table.dfimg
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD'].sort_values('scan', kind = 'mergesort')
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS'].sort_values('scan', kind = 'mergesort')
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            vis.output_instances_masks_to_tiff(output_path, df_pt_OD_ids, df_unique[scan], 'OD')
        if (len(df_pt_OS.index) > 0):
            vis.output_instances_masks_to_tiff(output_path, df_pt_OS_ids, df_unique[scan], 'OS')

def create_tif_output(mode = None):
    pred_file = os.path.join(output_path, 'coco_instances_results.json')
    dfimg_dummy = dataset_table.dfimg
    df_unique = dfimg_dummy.ptid.unique()
    vis = OutputVis(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file)
    for scan in range(len(df_unique)):
        df_currentpt = dfimg_dummy.loc[dfimg_dummy['ptid'] == df_unique[scan]]
        df_pt_OD = df_currentpt.loc[df_currentpt['eye'] == 'OD'].sort_values('scan', kind = 'mergesort')
        df_pt_OS = df_currentpt.loc[df_currentpt['eye'] == 'OS'].sort_values('scan', kind = 'mergesort')
        df_pt_OD_ids = df_pt_OD.index.values
        df_pt_OS_ids = df_pt_OS.index.values
        if (len(df_pt_OD.index) > 0):
            if (mode == 'bm'):
                vis.output_masks_to_tiff(output_path, df_pt_OD_ids, df_unique[scan], 'OD')
            elif (mode == 'bm-o'):
                vis.output_overlay_masks_to_tiff(output_path, df_pt_OD_ids, df_unique[scan], 'OD')
            elif (mode == 'im'):
                vis.output_instances_masks_to_tiff(output_path, df_pt_OD_ids, df_unique[scan], 'OD')
            else:
                print("No output mode selected!")
        if (len(df_pt_OS.index) > 0):
            if (mode == 'bm'):
                vis.output_masks_to_tiff(output_path, df_pt_OS_ids, df_unique[scan], 'OS')
            elif (mode == 'bm-o'):
                vis.output_overlay_masks_to_tiff(output_path, df_pt_OS_ids, df_unique[scan], 'OS')
            elif (mode == 'im'):
                vis.output_instances_masks_to_tiff(output_path, df_pt_OS_ids, df_unique[scan], 'OS')
            else:
                print("No output mode selected!")
def create_dfpts():
    if (dataset_table == None):
        create_table()
    dfpts = dataset_table.dfpts.sort_values(by=['dt_instances'],ascending=False)
    html_str = dfpts.style.format('{:.0f}').set_table_styles(styles).render()
    html_file = open(os.path.join(output_path, 'dfpts_'+dataset_name+'.html'),'w')
    html_file.write(html_str)
    html_file.close()

def create_dfimg():
    if (dataset_table == None):
        create_table()
    dfimg = dataset_table.dfimg.sort_index()
    html_str = dfimg.style.set_table_styles(styles).render()
    html_file = open(os.path.join(output_path, 'dfimg_'+dataset_name+'.html'),'w')
    html_file.write(html_str)
    html_file.close()

def main(args):
    parser = argparse.ArgumentParser(description='Run the detectron2 pipeline.')
    parser.add_argument('name', metavar = 'N', help='The name of your dataset.')
    parser.add_argument('input', metavar = 'I', help='The path to the directory containing your vol/dicom files.'  )
    parser.add_argument('output', metavar = 'O', help='The path to the folder where outputs will be stored.')
    parser.add_argument('--bm', action ='store_true', help='Output binary mask tif files.')
    parser.add_argument('--bmo', action ='store_true', help='Output binary mask overlay tif files.')
    parser.add_argument('--im', action ='store_true', help='Output instance mask overlay tif files.')
    parser.add_argument('--ptid', action ='store_true', help='Output a dataset html indexed by patient ids.')
    parser.add_argument('--imgid', action ='store_true', help='Output a dataset html indexed by image ids.')
    args = parser.parse_args(args)
    global dataset_name
    global output_path
    dataset_name = args.name
    data.dirtoextract = args.input
    data.extracteddir = args.output
    output_path = args.output
    if not os.path.isdir(output_path):
        print("Output dir does not exist! Making output dir...")
        os.mkdir(output_path)
    print("Processing input...")
    process_input()
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
    if args.bm:
        print("Creating binary masks tif (no overlay)...")
        create_tif_output(mode = 'bm')
    if args.bmo:
        print("Creating binary masks tif (with overlay)...")
        create_tif_output(mode = 'bm-o')
    if args.im:
        print("Creating instances masks tif (with overlay)...")
        create_tif_output(mode = 'im')
    if args.ptid:
        create_dfpts()
    if args.imgid:
        create_dfimg()
    print("Done!")

if __name__ == "__main__":
    main(sys.argv[1:])
    # main_alt()
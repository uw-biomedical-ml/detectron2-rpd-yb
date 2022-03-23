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

dpi= 120

def process_input(dataset_name = None, dirtoextract = None, output_path = None): # Processes input .vol files and creates the pk file.
    data.extractFiles(dataset_name = dataset_name, dirtoextract = dirtoextract, output_path = output_path)
    stored_data = data.rpd_data(dataset_name = dataset_name, output_path = output_path)
    pickle.dump(stored_data, open(os.path.join(data.script_dir,f"{dataset_name}_refined.pk"), "wb"))

def configure_model():
    cfg = get_cfg()
    moddir = os.path.dirname(os.path.realpath(__file__))
    name = 'working.yaml'
    cfg_path = os.path.join(moddir, name)
    cfg.merge_from_file(cfg_path)
    return cfg

def register_dataset(dataset_name = None):
    for name in [dataset_name]:
        try:
            DatasetCatalog.register(name, grab_dataset(name))
        except:
            print('Already registered.')
        MetadataCatalog.get(name).thing_classes = ["rpd"]

def run_prediction(cfg = None,dataset_name = None, output_path = None):
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

def run_ensemble(dataset_name = None, output_path = None):
    ens = Ensembler(output_path,dataset_name,["fold1", "fold2", "fold3", "fold4","fold5"],.2)
    ens.mean_score_nms()
    ens.save_coco_instances()
    return ens

def evaluate_dataset(dataset_name = None, output_path = None):
    myeval = EvaluateClass(dataset_name, output_path, iou_thresh = .2, prob_thresh=0.5,evalsuper=False)
    myeval.evaluate()
    with open(os.path.join(output_path,'scalar_dict.json'),"w") as outfile:
        json.dump(obj=myeval.summarize_scalars(),fp=outfile)
    return myeval

def create_table(myeval = None):
    dataset_table = CreatePlotsRPD.initfromcoco(myeval.mycoco,myeval.prob_thresh)
    dataset_table.dfimg.sort_index(inplace=True)
    return dataset_table
    #dataset_table.dfimg['scan'] = dataset_table.dfimg['scan'].astype('int') #depends on what we want scan field to be

def create_binary_masks_tif(dataset_name = None, output_path = None, dataset_table = None):
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

def create_binary_masks_overlay_tif(dataset_name = None, output_path = None, dataset_table = None):
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

def create_instance_masks_overlay_tif(dataset_name = None, output_path = None, dataset_table = None):
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

def create_tif_output(mode = None, dataset_name = None, output_path = None, dataset_table = None):
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
def create_dfpts(dataset_name = None, output_path = None, dataset_table = None):
    dfpts = dataset_table.dfpts.sort_values(by=['dt_instances'],ascending=False)
    html_str = dfpts.style.format('{:.0f}').set_table_styles(styles).render()
    html_file = open(os.path.join(output_path, 'dfpts_'+dataset_name+'.html'),'w')
    html_file.write(html_str)
    html_file.close()

def create_dfimg(dataset_name = None, output_path = None, dataset_table = None):
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
    name = args.name
    input = args.input
    output = args.output
    if not os.path.isdir(output):
        print("Output dir does not exist! Making output dir...")
        os.mkdir(output)
    print("Processing input...")
    process_input(dataset_name = name, dirtoextract = input, output_path = output)
    print("Configuring model...")
    cfg = configure_model()
    print("Registering dataset...")
    register_dataset(dataset_name = name)
    print("Running inference...")
    run_prediction(cfg = cfg, dataset_name = name, output_path = output)
    print("Running ensemble...")
    run_ensemble(dataset_name = name, output_path = output)
    print("Evaluating dataset...")
    eval = evaluate_dataset(dataset_name = name, output_path = output)
    print("Creating dataset table...")
    table = create_table(myeval = eval)
    if args.bm:
        print("Creating binary masks tif (no overlay)...")
        create_tif_output(mode = 'bm', dataset_name = name, output_path = output, dataset_table = table)
    if args.bmo:
        print("Creating binary masks tif (with overlay)...")
        create_tif_output(mode = 'bm-o', dataset_name = name, output_path = output, dataset_table = table)
    if args.im:
        print("Creating instances masks tif (with overlay)...")
        create_tif_output(mode = 'im', dataset_name = name, output_path = output, dataset_table = table)
    if args.ptid:
        create_dfpts(dataset_name = name, output_path = output, dataset_table = table)
    if args.imgid:
        create_dfimg(dataset_name = name, output_path = output, dataset_table = table)
    print("Done!")

if __name__ == "__main__":
    main(sys.argv[1:])
    # main_alt()
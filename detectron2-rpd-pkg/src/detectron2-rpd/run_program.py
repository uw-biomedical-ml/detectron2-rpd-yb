import datasets.data as data
import pickle
from detectron2.config import get_cfg
from analysis_lib import grab_dataset
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from Ensembler import Ensembler
from analysis_lib import EvaluateClass,CreatePlotsRPD,OutputVis
import logging
logging.basicConfig(level=logging.INFO)

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

def process_input(dataset_name, dirtoextract, extracted_path): # Processes input .vol files and creates the pk file.
    data.extractFiles(dataset_name, dirtoextract, extracted_path)
    stored_data = data.rpd_data(dataset_name, extracted_path)
    pickle.dump(stored_data, open(os.path.join(data.script_dir,f"{dataset_name}.pk"), "wb"))

def configure_model():
    cfg = get_cfg()
    moddir = os.path.dirname(os.path.realpath(__file__))
    name = 'working.yaml'
    cfg_path = os.path.join(moddir, name)
    cfg.merge_from_file(cfg_path)
    return cfg

def register_dataset(dataset_name):
    for name in [dataset_name]:
        try:
            DatasetCatalog.register(name, grab_dataset(name))
        except:
            print('Already registered.')
        MetadataCatalog.get(name).thing_classes = ["rpd"]

def run_prediction(cfg, dataset_name, output_path):
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

def run_ensemble(dataset_name, output_path, iou_thresh = 0.2):
    ens = Ensembler(output_path,dataset_name,["fold1", "fold2", "fold3", "fold4","fold5"],  iou_thresh=iou_thresh)
    ens.mean_score_nms()
    ens.save_coco_instances()
    return ens

def evaluate_dataset(dataset_name, output_path,iou_thresh = 0.2,prob_thresh = 0.5):
    myeval = EvaluateClass(dataset_name, output_path, iou_thresh = iou_thresh, prob_thresh=prob_thresh,evalsuper=False)
    myeval.evaluate()
    with open(os.path.join(output_path,'scalar_dict.json'),"w") as outfile:
        json.dump(obj=myeval.summarize_scalars(),fp=outfile)
    return myeval

def create_table(myeval):
    dataset_table = CreatePlotsRPD.initfromcoco(myeval.mycoco,myeval.prob_thresh)
    dataset_table.dfimg.sort_index(inplace=True)
    return dataset_table
    #dataset_table.dfimg['scan'] = dataset_table.dfimg['scan'].astype('int') #depends on what we want scan field to be

def output_vol_predictions(dataset_table,vis,volID,output_path,output_mode='pred_overlay'):
    dfimg = dataset_table.dfimg
    imgids = dfimg[dfimg.volID ==volID].sort_index().index.values
    outname = os.path.join(output_path,f'{volID}.tiff')
    if output_mode=='pred_overlay':
        vis.output_pred_to_tiff(imgids,outname,pred_only=False)
    elif output_mode == 'pred_only':
        vis.output_pred_to_tiff(imgids,outname,pred_only=True)
    elif output_mode == 'originals':
        vis.output_ori_to_tiff(imgids,outname)
    elif output_mode == 'all':
        vis.output_all_to_tiff(imgids,outname)
    else:
        print(f'Invalid mode {output_mode} for function output_vol_predictions.')

def output_dataset_predictions(dataset_table,vis,output_path,output_mode = 'pred_overlay',draw_mode='default'):
    vis.set_draw_mode(draw_mode)
    os.makedirs(output_path,exist_ok=True) ## should this be set to exist_ok=False?
    for volID in dataset_table.dfvol.index:
        output_vol_predictions(dataset_table,vis,volID,output_path,output_mode)

def create_dfvol(dataset_name, output_path, dataset_table):
    dfvol = dataset_table.dfvol.sort_values(by=['dt_instances'],ascending=False)
    html_str = dfvol.style.format('{:.0f}').set_table_styles(styles).render()
    html_file = open(os.path.join(output_path, 'dfvol_'+dataset_name+'.html'),'w')
    html_file.write(html_str)
    html_file.close()

def create_dfimg(dataset_name, output_path, dataset_table):
    dfimg = dataset_table.dfimg.sort_index()
    html_str = dfimg.style.set_table_styles(styles).render()
    html_file = open(os.path.join(output_path, 'dfimg_'+dataset_name+'.html'),'w')
    html_file.write(html_str)
    html_file.close()

def main(args):
    parser = argparse.ArgumentParser(description='Run the detectron2 pipeline.')
    parser.add_argument('name', metavar = 'N', help='The name of your dataset.')
    parser.add_argument('input', metavar = 'I', help='The path to the directory containing your vol/dicom files.'  )
    parser.add_argument('extracted', metavar = 'E', help='The path to the directory where extracted images will be stored.')
    parser.add_argument('output', metavar = 'O', help='The path to the directory where model predictions and other data will be stored.')
    parser.add_argument('--bm', action ='store_true', help='Output binary mask tif files.')
    parser.add_argument('--bmo', action ='store_true', help='Output binary mask overlay tif files.')
    parser.add_argument('--im', action ='store_true', help='Output instance mask overlay tif files.')
    parser.add_argument('--volid', action ='store_true', help='Output a dataset html indexed by vol ids.')
    parser.add_argument('--imgid', action ='store_true', help='Output a dataset html indexed by image ids.')
    args = parser.parse_args(args)

    name = args.name
    input = args.input
    extracted = args.extracted
    output = args.output
    iou_thresh = 0.2
    prob_thresh = 0.5
    if not os.path.isdir(extracted):
        print("Extracted dir does not exist! Making extracted dir...")
        os.mkdir(extracted)
    if not os.path.isdir(output):
        print("Output dir does not exist! Making output dir...")
        os.mkdir(output)
    print("Processing input...")
    process_input(name, input, extracted)
    print("Configuring model...")
    cfg = configure_model()
    print("Registering dataset...")
    register_dataset(name)
    print("Running inference...")
    run_prediction(cfg, name, output)
    print("Running ensemble...")
    run_ensemble(name, output, iou_thresh)
    print("Evaluating dataset...")
    eval = evaluate_dataset(name, output, iou_thresh, prob_thresh)
    print("Creating dataset table...")
    table = create_table(eval)
    vis = OutputVis(name,
        prob_thresh = eval.prob_thresh, 
        pred_mode = 'file',
        pred_file = os.path.join(output, 'coco_instances_results.json'),
        has_annotations=False)
    vis.scale=1.0
    if args.bm:
        print("Creating binary masks tif (no overlay)...")
        vis.annotation_color='w'
        output_dataset_predictions(table,vis,os.path.join(output,'predicted_binary_masks'),'pred_only','bw')
    if args.bmo:
        print("Creating binary masks tif (with overlay)...")
        output_dataset_predictions(table,vis,os.path.join(output,'predicted_binary_overlays'),'pred_overlay','bw')
    if args.im:
        print("Creating instances masks tif (with overlay)...")
        output_dataset_predictions(table,vis,os.path.join(output,'predicted_instance_overlays'),'pred_overlay','default')
    if args.volid:
        create_dfvol(name, output, table)
    if args.imgid:
        create_dfimg(name, output, table)
    print("Done!")

if __name__ == "__main__":
    main(sys.argv[1:])
    # main_alt()
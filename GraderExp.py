from detectron2.data import DatasetCatalog,MetadataCatalog
from plain_train_net import grab_dataset, OutputVis
import os
import matplotlib.pyplot as plt
plt.style.use('ybpres.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import numpy as np
import random
import pandas as pd


class GraderExperiment(OutputVis):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

    def output_to_pdf(self,ImgIds,outname,rng=np.random.RandomState()):
        idx1 = rng.choice([1, 2],size=len(ImgIds)) #randomly assign 1 or 2 to idx1 array
        idx2 = [{1:2,2:1}[i] for i in idx1] #if idx1 is 1 idx2 is 2, etc...
        #idx1 is panel index of ground truth annotations
        df_answers = pd.DataFrame(np.vstack((ImgIds,idx1)).T,columns=['ImgId','panel_index'])

        with PdfPages(outname) as pdf:
            for i,imgid in enumerate(tqdm(ImgIds)):
                img, img_model = self.get_image(imgid)# annotated images
                img_ori = self.get_ori_image(imgid)

                crop_range = self.height_crop_range(np.array(img.convert('L')),height_target=256*3)
                img = np.array(img)[crop_range]
                img_model = np.array(img_model)[crop_range]
                img_ori = np.array(img_ori)[crop_range]

                fig, ax = plt.subplots(3,1,figsize=[22,15],dpi=200)
                ax[0].imshow(img_ori)
                ax[0].set_title(f'{i} Original Image')
                ax[0].set_axis_off()
                
                ax[idx1[i]].imshow(img) #ground truth
                ax[idx1[i]].set_title(f'{i} Prediction')
                ax[idx1[i]].set_axis_off()

                ax[idx2[i]].imshow(img_model) #
                ax[idx2[i]].set_title(f'{i} Prediction')
                ax[idx2[i]].set_axis_off()

                pdf.savefig(fig)
                plt.close(fig)
        return df_answers

def get_balanced_imgids(df,shuffle=True, n=25,rng=None):
    """Randomly choose 2*n image ids from df such that the number of over-segmented images match the number of under-segmented images.

    Args:
        df (pd.DataFrame): RPDPlt.dfimg object. Dataframe to sample from. 
        shuffle (bool, optional): Shuffle image Ids. Defaults to True.
        n (int, optional): Number of samples to take from each side. Defaults to 25.

    Returns:
        list[str]: list of image ids
    """
    dfsort = (df.gt_instances - df.dt_instances).sort_values(ascending=False)
    dfsort = dfsort.to_frame('del_inst').reset_index()
    idx_m = dfsort[dfsort['del_inst']<0].index[0]
    idx_p = dfsort[dfsort['del_inst']>0].index[-1]
    imgids = pd.concat([dfsort.iloc[0:idx_p].sample(n,random_state=rng),dfsort.iloc[idx_m:].sample(n,random_state=rng)]).set_index('index')
    if shuffle:
        imgids = imgids.sample(frac=1,random_state=rng)
    return imgids.index.values


dataset_name = 'val'
#rng = np.random.default_rng(1234) #newer way 
rng = np.random.RandomState(2345) #compatable with older/current pandas

for name in [dataset_name]:
    try:
        DatasetCatalog.register(name, grab_dataset(name))
    except:
        print('Already registered.')
        #do nothing
    MetadataCatalog.get(name).thing_classes = ["rpd"]



pred_file = "output_"+ dataset_name + "/coco_instances_results.json"
out_file = os.path.join("output_"+ dataset_name,'GraderExperiment_'+dataset_name+'.pdf')
ans_file = os.path.join("output_"+ dataset_name,'GraderExperiment_'+dataset_name+'GTpanelindex.csv')

df = pd.read_csv(os.path.join("output_"+ dataset_name,'dfimg_val.csv'),index_col=0)
#ImgIds = np.abs(df.gt_instances-df.dt_instances).sort_values(ascending=False).iloc[0:50].sample(frac=1).index.values
ImgIds = get_balanced_imgids(df,rng=rng,n=25)
dfout = df.loc[ImgIds] #images chosen for experiment
print(dfout.gt_instances - dfout.dt_instances) #looking at instance count balance
dfwork = pd.DataFrame([np.array(a) for a in np.char.split(np.array(dfout.index.values,dtype=str),'_')],columns = ['ptid','eye','scan'])
print('summary: ', dfwork.nunique())

vis = GraderExperiment(dataset_name,prob_thresh = 0.5,pred_mode='file',pred_file=pred_file,has_annotations=True,draw_mode='bw')
df_answers = vis.output_to_pdf(ImgIds,out_file,rng=rng)
print(df_answers)
df_answers.to_csv(ans_file)

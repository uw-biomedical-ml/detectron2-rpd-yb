from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import warnings
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../ybpres.mplstyle')

def Wilson_CI(p,n,z):
    if (p<0 or p>1 or n==0):
        if (p<0 or p>1):
            warnings.warn(f'The value of proportion {p} must be in the range [0,1]. Returning identity for CIs.')
        else:
            warnings.warn(f'The number of counts {n} must be above zero. Returning identity for CIs.')
        return (p,p)
    sym = z*(p*(1-p)/n + z*z/4/n/n)**.5
    asym = p + z*z/2/n
    fact = 1/(1+z*z/n)
    upper = fact*(asym+sym)
    lower = fact*(asym-sym)
    return (lower,upper)

class EvaluateClassGraders():  
    def __init__(self,annotation_file_gt, annotation_file_dt,prob_thresh=0.5,iou_thresh = 0.1,evalsuper=True):
        self.cocoGt = COCO(annotation_file_gt)
        self.cocoDt = COCO(annotation_file_dt)
        self.mycoco = COCOeval(self.cocoGt,self.cocoDt,iouType ='segm')
        self.prob_thresh = prob_thresh #instance probabilty threshold for scalars (precision,recall,fpr for scans)
        self.iou_thresh = iou_thresh #iou threshold for defining precision,recall
        self.pr = None
        self.rc = None
        self.fpr = None
    def evaluate(self):

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
                        if (np.array(el['dtScores']) >=s).sum() > 0: #if at least one score over threshold for recall level
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
        f1 = 2*(p*r)/(p + r)
        fpr = self.get_fpr()

        #Confidence intervals
        z=1.96 #95% Gaussian
        #instance count 
        inst_cnt = self.count_instances()
        n_r = inst_cnt['gt_instances']
        n_p = inst_cnt['dt_instances']
        n_fpr = inst_cnt['gt_neg_scans']
        
        def stat_CI(p,n,z):
            return z*np.sqrt(p*(1-p)/n)

        r_ci = Wilson_CI(r,n_r,z) 
        p_ci = Wilson_CI(p,n_p,z)
        fpr_ci = Wilson_CI(fpr,n_fpr,z)

        #propogate errors for f1
        int_r = stat_CI(r,n_r,z)
        int_p = stat_CI(p,n_p,z)
        int_f1 =(f1)*np.sqrt(int_r**2 * (1/r - 1/(p+r))**2 + int_p**2 * (1/p - 1/(p+r))**2)
        f1_ci = (f1-int_f1,f1+int_f1)

        dd = dict(precision = float(p),precision_ci = p_ci,recall=float(r), recall_ci = r_ci,f1 = float(f1),f1_ci = f1_ci, fpr=float(fpr),fpr_ci = fpr_ci, iou=self.iou_thresh,probability=self.prob_thresh)
        return dd

    def count_instances(self):
        gt_inst = 0
        dt_inst = 0
        gt_neg_scans = 0
        for key,val in self.cocoGt.imgs.items():
            imgid = val['id']
            #Gt instances
            annIdsGt = self.cocoGt.getAnnIds([imgid])
            annsGt = self.cocoGt.loadAnns(annIdsGt)
            gt_inst+=len(annsGt)
            if len(annsGt)==0:
                gt_neg_scans+=1

            #Dt instances
            annIdsDt = self.cocoDt.getAnnIds([imgid])
            annsDt = self.cocoDt.loadAnns(annIdsDt)
            annsDt = [ann for ann in annsDt if ann['score']>self.prob_thresh]
            dt_inst+=len(annsDt)

        return dict(gt_instances=gt_inst,dt_instances=dt_inst,gt_neg_scans = gt_neg_scans)

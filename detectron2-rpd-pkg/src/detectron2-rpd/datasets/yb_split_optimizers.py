import numpy as np
import pandas as pd


from itertools import product,islice
from time import time
from tqdm import tqdm


def L1(T,Y,axis=(1,2)):
    E = np.sum(np.abs(T-Y),axis=axis)
    return E
def L2(T,Y,axis=(1,2)):
    E = np.sum((T-Y)**2,axis=axis)
    return E
def findbestcombo(X,T,k=2,errfunc=L1):
        #num_classes = 3
        gen = product(range(k+1),repeat=X.shape[-1]) #generator
        a = np.array(list(gen)) # all possible group combinations for patients
        #convert to one hot where last channel is combination
        a_off = a - a.min()
        #onehot matrix of all possible train/val/test patient-level splits
        W = (np.arange(k+1) == a_off[...,None]).astype(int) 
        #compute the cost function
        Y = X@W #train/val/test split pixel counts for all possible splits
        E = errfunc(T,Y)
        ind = np.argmin(E) #index of split with minimum error
        return W[ind],E[ind]
def getSplitScalablekfold(df, k=2,rtest=.2,err=[],err_class=[],err_size=[],batch_size=5,iters=1000,errtype='L1',size_err_weight=1,class_err_weight=1,mode=1):
    '''
    Data splitter. Reads in a dataframe containing group/ptid labels and pixel stats for each image. Splitter assigns scans to a test fold and k other folds, splitting along groups.
    This splitter implicitly optimizes for proportional positive class pixel distributions across folds. Splitter explicitly optimizes for target sample sizes across folds and proportional positive class pixel distributions across folds based on errtype and error weight factors. 
    
    Args:
    df (pd.DataFrame): dataframe containing group/ptid labels and pixels stats (yellow, white, red) for each image.
    k (int): number of folds excluding the test set
    rtest (float, optional): ratio of data to put into test set. Defaults to 0.2.
    iters (int, optional): number of iterations to try. Defaults to 1000.
    batch_size (int,optional): number of groups per batch. Use smaller batch_size to avoid computational blow-up. Defualts to 5.
    size_err_weight (int, optional): weight factor of the error corresponding to the number of samples per fold objective
    class_err_weight (int, optional): weight factor of the error corresponding to the number of positive class pixels per fold objective
    err (list, optional): list to append total error to
    err_class (list, optional): list to append pixel count error to
    err_size (list, optional): list to append sample size error to
    errtype (str,optional): type of error to minimize. Default is 'L1'.
    mode (int, optional): mode 1 minimizes weighted sum of the class pixel and sample size errors. mode 2 minimizes sample size error first and then class pixel error. Default is mode 1.

    Returns:
        df (pd.DataFrame): a dataframe containing the fold assignment of each patient eye
        dfsummary (pd.DataFrame) : a dataframe summarizing the folds
    '''
    start = time()
    grp = df[['ptid','yellow','white','red']].groupby('ptid').sum()
    #grp = grp.iloc[0:15] #for debugging !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #total number of class pixels present in data set
    totals = grp.sum()
    num_classes = 3
    #target matrix, rows are classes. columns are train, val, and test sets
    split_ratios = np.vstack((np.tile((1-rtest)/k,(k,1)),rtest))
    grp_weights = ((df.groupby('ptid').size()/len(df)).values)[np.newaxis,:,np.newaxis] #based on number of samples per group/ptid, [iter,ptid,folds]

    T = (split_ratios*totals.values).T #global target
    X = grp.T.values # rows are classes, columns are ptids
    X = X[np.newaxis,:,:] # add dimension for 3D matrix multiplication
    nbatches = X.shape[-1]//batch_size +1
    w = np.zeros((iters,X.shape[-1],k+1))
    W = np.zeros((iters,X.shape[-1],k+1))
    y = np.zeros((iters,num_classes,k+1))
    #start outer loop here, shuffled patients
    if errtype=='L1':
        errfunc = L1
        size_norm = 1
    elif errtype=='L2':
        errfunc = L2
        size_norm=len(df)
    else: 
        print('Invalid errtype {s}. Only L1 or L2 permitted. Using L1')
        errfunc = L1
    #np.random.seed(333)
    for j in tqdm(range(iters)):
        ind_shuffle = np.random.permutation(X.shape[-1]) #shuffled indices
        #ind_shuffle = range(X.shape[-1])#no shuffle
        x = X[:,:,ind_shuffle] #shuffled patient counts
        w0 = np.zeros((X.shape[-1],k+1)) #best combo for this iteration
        for i in range(nbatches):
            #print('Optimizing batch {}'.format(i))
            
            if (i+1 ==nbatches): #last batch
                if x.shape[-1]%batch_size !=0: #data isn't multiple of batch
                    x_chunk = x[:,:,(i*batch_size):]
            else:
                x_chunk = x[:,:,(i*batch_size):((i+1)*batch_size)]
            
            local_totals = x_chunk[0].sum(axis=1) #totals for this batch
            previous_totals = x[0,:,:((i)*batch_size)].sum(axis=1) #totals for all previous batches
            local_target = (split_ratios*local_totals).T #batch target
            previous_target = (split_ratios*previous_totals).T #target for all previous batches
            achieved_count = x[0,:,:(i*batch_size)]@w0[:i*batch_size] #achieved count for all previous batches 
            currentT = local_target + previous_target - achieved_count # new target for curent batch
            
            
            #pdb.set_trace()
            if (i+1 ==nbatches): #last batch
                if x.shape[-1]%batch_size !=0: #data isn't multiple of batch
                    w0[i*batch_size:], _ = findbestcombo(x_chunk,currentT,k,errfunc)
            else:
                w0[i*batch_size:(i+1)*batch_size], _ = findbestcombo(x_chunk,currentT,k,errfunc)
        w[j] = w0.copy()
        W[j,ind_shuffle] = w0.copy() #unscrambled
        y[j] = x@w0
                

    e_class = errfunc(T,y)/T.sum() #remaining dimension is iter version
    W_fold = (W*grp_weights).sum(axis=1) #sum over groups in each fold
    e_size = errfunc(split_ratios.T,W_fold,axis=-1)*size_norm #sum over folds in each iteration, already normalized 
    e = class_err_weight*e_class + size_err_weight*e_size
    err.append(e) #update optional reference arg
    err_class.append(e_class)
    err_size.append(e_size)
    if mode==1:
        ind = np.argmin(e) #index of split with minimum error
    elif mode==2:
        ind = np.where(np.abs(e_size - np.min(e_size))<1e-4)
        ind2 = np.argmin(e_class[ind])
        ind = ind[0][ind2]
    else:
        print('Error: imporoper mode {}'.format(mode))
        exit(1) 
    W0 = W[ind]
    print('Best err: {}'.format(e[ind]))
    print('Err min = ', e[ind], 'E_sample = ',e_size[ind],'E_class = ',e_class[ind])
    Y = X[0]@W0
    E_class = errfunc(T,Y,axis=(0,1))/T.sum()
    E_size = errfunc(split_ratios.T[0],(W0*grp_weights[0]).sum(axis=0),axis=0)*size_norm
    E0 = class_err_weight*E_class + size_err_weight*E_size

    end = time()    
    print('\nBest split:')
    colnames = ['fold'+str(i) for i in range(1,k+1)] + ['test']
    print(pd.DataFrame(W0,index = grp.index,columns=colnames))
    print('\nTarget pixel distributions:')
    print(pd.DataFrame(T,index=totals.index,columns=colnames))
    print('\nAchieved pixel distributions:')
    print(pd.DataFrame((X @ W0)[0],index=totals.index,columns=colnames))
    print('Best err: {}'.format(E0))
    print('Err min = ', E0, 'E_sample = ',E_size,'E_class = ',E_class)
    
    df = df.set_index('ptid')
    df = df.assign(fold = None)
    for i in range(k+1):
        idx = grp[W0[:,i].astype(bool)].index
        df.loc[idx,'fold'] = colnames[i]
    
    grp = df.groupby('fold')
    grp2 = df.reset_index().groupby('fold')
    grp3 = df.reset_index().groupby(['fold','ptid','eye']).size().groupby('fold')
    dfsummary = pd.concat([grp2[['scan','ptid']].nunique(),grp3.size().to_frame('eye'),grp.sum()],axis=1)
    print('-------------------------------')
    print(dfsummary)
 
    
    # train_list = list(df_train.scan) 
    # val_list = list(df_val.scan) 
    # test_list = list(df_test.scan) 
    
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return df,dfsummary
    #return train_list,val_list,test_list

def getSplitkfold(df,k,rtest = .2, iters = 1000,err=[],err_class=[],err_size=[],size_err_weight=1,class_err_weight=1,mode=1,errtype='L1'):
    """Data splitter. Reads in a dataframe containing group/ptid labels and pixel stats for each image. Splitter assigns scans to a test fold and k other folds, splitting along groups.
    This splitter implicitly optimizes for equal split ratios among the k folds. Splitter explicitly optimizes for target sample sizes across folds and proportional positive class pixel distributions across folds based on errtype and error weight factors. 

    Args:
        df (pd.DataFrame): dataframe containing group/ptid labels and pixels stats (yellow, white, red) for each image.
        k (int): number of folds excluding the test set
        rtest (float, optional): ratio of data to put into test set. Defaults to 0.2.
        iters (int, optional): number of iterations to try. Defaults to 1000.
        size_err_weight (int, optional): weight factor of the error corresponding to the number of samples per fold objective
        class_err_weight (int, optional): weight factor of the error corresponding to the number of positive class pixels per fold objective
        err (list, optional): list to append total error to
        err_class (list, optional): list to append pixel count error to
        err_size (list, optional): list to append sample size error to
        mode (int, optional): mode 1 minimizes weighted sum of the class pixel and sample size errors. mode 2 minimizes sample size error first and then class pixel error. Default is mode 1.
        errtype (str,optional): type of error to minimize. Default is 'L1'.
        

    Returns:
        df (pd.DataFrame): a dataframe containing the fold assignment of each patient eye
        dfsummary (pd.DataFrame) : a dataframe summarizing the folds
    """
    start = time()
    df = df.set_index('ptid')
    class_totals = df[['yellow','white','red']].sum()
    split_ratios = np.vstack((np.tile((1-rtest)/k,(k,1)),rtest))
    T = (split_ratios*class_totals.values) #global target
    if T.sum()==0:
        class_norm = 0
    else:
        class_norm = 1/T.sum()

    target = [(1-rtest)/k]*k + [rtest]
    t_cum = np.cumsum(target)
    t_cum = np.round(t_cum,9) #avoid rounding issues to 1e-9 precision
    scan_target = np.array(target)*len(df)

    ptwt = df.reset_index().groupby(['ptid','eye']).size().groupby(level=0).size()
    ptwt = ptwt/ptwt.sum() #Series
    folds = pd.DataFrame(index = ptwt.index)
    folds_final = None

    foldnames = ['fold'+str(i) for i in range(1,k+1)] + ['test']
    E_size = np.empty(iters)
    E_class = np.empty(iters)
    E = np.empty(iters)
    Emin = np.inf
    Emin_class = np.inf
    Emin_size = np.inf

    if errtype=='L1':
        errfunc = L1
    elif errtype=='L2':
        errfunc = L2
    else: 
        print('Invalid errtype {s}. Only L1 or L2 permitted. Using L1')
        errfunc = L1
    for j in tqdm(range(iters)):
        folds.assign(fold=None)        
        ptwt_cum = ptwt.sample(frac=1).cumsum()   
        ptwt_cum = np.round(ptwt_cum,9) #avoid rounding issues to 1e-9 precision     
        folds.loc[ptwt_cum[ptwt_cum<=t_cum[0]].index,'fold'] = foldnames[0] #first fold
        for i in range(0,k):
            folds.loc[ptwt_cum[(ptwt_cum>t_cum[i])&(ptwt_cum<=t_cum[i+1])].index, 'fold'] = foldnames[i+1]
        #pdb.set_trace()
        dm = df.merge(folds,left_index=True,right_index=True).groupby('fold')
        E_size[j] = errfunc(scan_target[np.where(scan_target)],dm.size(),axis=0)/scan_target.sum()
        E_class[j] = errfunc(T[np.where(scan_target)],np.array(dm[['yellow','white','red']].sum()),axis=(0,1))*class_norm
        E[j] = size_err_weight*E_size[j] + class_err_weight*E_class[j]
        #pdb.set_trace()
        if mode==1:
            if E[j]<Emin:
                folds_final = folds.copy()
                Emin = E[j]
                Emin_size = E_size[j]
                Emin_class = E_class[j]
            else:
                continue
        elif mode==2:
            if Emin_size>E_size[j]:
                folds_final = folds.copy()
                Emin = E[j]
                Emin_size = E_size[j]
                Emin_class = E_class[j] 
            elif E_size[j]==Emin_size and E_class[j]<Emin_class:
                folds_final = folds.copy()
                Emin = E[j]
                #Emin_size = E_size[j]
                Emin_class = E_class[j]
            else:
                continue
        else:
            print('Error: imporoper mode {}'.format(mode))
            os.exit(1)        

    df = df.merge(folds_final,left_index=True,right_index=True)
    err.append(E)
    err_size.append(E_size)
    err_class.append(E_class)
    grp = df.groupby('fold')
    grp2 = df.reset_index().groupby('fold')
    grp3 = df.reset_index().groupby(['fold','ptid','eye']).size().groupby('fold')
    dfsummary = pd.concat([grp2[['scan','ptid']].nunique(),grp3.size().to_frame('eye'),grp.sum()],axis=1)
    print('-------------------------------')
    print(dfsummary)
    print('Err min = ', Emin, 'E_sample = ',Emin_size,'E_class = ',Emin_class)
    
    return df,dfsummary
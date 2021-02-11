import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from pathlib import Path
from refactor.utils.tree_helpers import get_merged_ordered_classes
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

from sklearn.model_selection import StratifiedKFold
from refactor.utils.compute import contingency
from refactor.utils.plots import matrix_scatterplot

representation_pth = '/home/rohan/Remote-AI/dat/result/TE_NM/'
origdata_pth = Path('./refactor/data/proc/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat')
E_names_file = Path('./refactor/data/proc/E_names.json')
figure_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/doc/Patchseq_NM_2020/'

O = sio.loadmat(origdata_pth,squeeze_me=True)
O['E_dat']=np.concatenate([O['E_pc_scaled'],O['E_feature']],axis = 1)
O['E_names']=np.concatenate([O['pc_name'],O['feature_name']],axis = 0)
with open(E_names_file) as f:
    temp = json.load(f)
O['E_names'] = np.array([temp[i] for i in O['E_names']])

#Get t-types in order as per reference taxonomy:
O = sio.loadmat(origdata_pth,squeeze_me=True)
n_required_classes = np.unique(O['cluster']).size
t_type_labels,t_types = get_merged_ordered_classes(data_labels=O['cluster'].copy(),
                                                   htree_file='./refactor/data/proc/dend_RData_Tree_20181220.csv',
                                                   n_required_classes=n_required_classes,
                                                   verbose=False)

#well-sampled t-types:
min_sample_thr=10
t_types_well_sampled = []
for t in t_types:
    if np.sum(O['cluster']==t)>min_sample_thr:
        t_types_well_sampled.append(t)
        
for n_ref_classes in range(5,53,1):
    ref_tax_33,ref_tax_33_order = get_merged_ordered_classes(np.array(t_types_well_sampled).copy(),
                                                    htree_file='./refactor/data/proc/dend_RData_Tree_20181220.csv',
                                                    n_required_classes=n_ref_classes,
                                                    verbose=False)

    #Get 33 type labels for individual samples
    gt_labels = O['cluster'].copy()
    for (orig,updated) in zip(t_types_well_sampled,ref_tax_33):
        gt_labels[gt_labels==orig]=updated
        
    keep = np.isin(gt_labels,ref_tax_33)
    keep_ind = np.flatnonzero(keep)

    #Uncoupled representations
    alpha_T=1.0
    alpha_E=1.0
    lambda_TE=0.0
    aug = 0
    n_cvfolds = 10

    gt_labels=gt_labels[keep_ind]
    CVdict={}
    key_list = ['zE','zT']
    for cv in range(n_cvfolds):
        cvfold_fname=(f'NM_Edat_pcipfx_aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}'+\
                f'_ad_{aug:d}_ld_3_bs_200_se_500_ne_1500_cv_{cv:d}_ri_0_500_ft-summary').replace('.','-') + '.mat'

        if Path(representation_pth+cvfold_fname).is_file(): 
            X = sio.loadmat(representation_pth+cvfold_fname,squeeze_me=True)
            CVdict[cv] = {key:X[key][keep_ind,:] for key in key_list}
            del X       
        else: print(cvfold_fname,'not found')
            
    XE = np.concatenate([O['E_pc_scaled'],O['E_feature']],axis = 1)
    XE[np.isnan(XE)]=0
    XE = XE[keep_ind,:]

    XT = O['T_dat'].copy()
    XT = XT[keep_ind,:]

    skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=True)
    ind_dict = [{'train':train_ind, 'val':val_ind} for train_ind, val_ind in skf.split(X=np.zeros(shape=gt_labels.shape), y=gt_labels)]

    cv = 0 #Pick a single representation
    f_co = []
    acc_t_e_gt_list = []
    for fold in [0,1,2,3,4,5]:
        qda = QDA(reg_param=1e-2,store_covariance=True)
        train_ind = ind_dict[fold]['train']
        val_ind = ind_dict[fold]['val']
        
        qda.fit(CVdict[cv]['zE'][train_ind],gt_labels[train_ind])
        lbl_train_pred_E = qda.predict(CVdict[cv]['zE'][train_ind])
        lbl_val_pred_E   = qda.predict(CVdict[cv]['zE'][val_ind])

        qda.fit(CVdict[cv]['zT'][train_ind],gt_labels[train_ind])
        lbl_train_pred_T = qda.predict(CVdict[cv]['zT'][train_ind])
        lbl_val_pred_T   = qda.predict(CVdict[cv]['zT'][val_ind])
        
        #Consensus between E and T supervised classification labels
        C = contingency(a=lbl_val_pred_T,
                        b=lbl_val_pred_E,
                        unique_a=ref_tax_33_order,
                        unique_b=ref_tax_33_order)
        
        f_co.append(np.sum(np.diag(C>0))/C.shape[0])
        #print(f'frac_co_occupied = {f_co[-1]:0.3f}')

        ari = adjusted_rand_score(lbl_val_pred_T,lbl_val_pred_E)
        ami = adjusted_mutual_info_score(lbl_val_pred_T,lbl_val_pred_E)

        all_gt = np.concatenate([gt_labels[train_ind],gt_labels[val_ind]])
        all_E = np.concatenate([lbl_train_pred_E,lbl_val_pred_E])
        all_T = np.concatenate([lbl_train_pred_T,lbl_val_pred_T])             
        acc_t_e = np.sum(all_E==all_T )/all_T.size
        acc_t_gt = np.sum(all_T==all_gt)/all_T.size
        acc_t_e_gt = np.sum(np.logical_and(all_T==all_E, all_T==all_gt))/all_T.size
        acc_t_e_gt_list.append(acc_t_e_gt)
        #print(f'ari:{ari:0.3f}  ami:{ami:0.3f}' + f'acc_t_e:{acc_t_e:0.2f}  acc_t_gt:{acc_t_gt:0.2f}  acc_t_e_gt:{acc_t_e_gt:0.2f}')

    print(f'{np.unique(ref_tax_33).size} classes, avg accuracy: {np.mean(acc_t_e_gt_list):0.3f} + {np.std(acc_t_e_gt_list):0.3f},  avg_fco: {np.mean(f_co):0.3f} + {np.std(f_co):0.3f}')
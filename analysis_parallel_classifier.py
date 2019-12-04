#python -m analysis_parallel_classifier --cvfold 0 --alpha_T 0.0  --alpha_E 1.0 --lambda_TE 1.0 --root_node n88 --start_i 0  --stop_i 1000 --embedding zE --rand_seed 0 &
#python -m analysis_parallel_classifier --cvfold 0 --alpha_T 0.0  --alpha_E 1.0 --lambda_TE 1.0 --root_node n88 --start_i 1000  --stop_i 2000 --embedding zE --rand_seed 0 &
#python -m analysis_parallel_classifier --cvfold 0 --alpha_T 1.0  --alpha_E 0.5 --lambda_TE 1.0 --root_node n88 --start_i 1000  --stop_i 2000 --embedding zE --rand_seed 0

import csv
import json
import argparse
import sys
import pdb

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import adjusted_rand_score
from timebudget import timebudget

from analysis_tree_helpers import HTree


def relabel_restrict_inputs(CV=None,O=None,this_classification=[],descendant_dict={}):
    """Restrict the train, validation and test subsets to relevant portion of the tree, and relabel according to current classification being considered.
    
    Arguments:
        CV -- dict with one of the cross validation sets, with fields `zT`, `zE`, `train_ind`,`val_ind`,`test_ind`
        O -- dict with annotation field `cluster`
        this_classification -- list of labels remaining in the classification
        descendant_dict -- dict with descendants based on the tree
    
    Returns:
        X -- dictionary with train,val and test subsets to train the classifiers.
    """

    X={}
    #Initialize
    for ds in ['train','val','test']:
        X[ds]={}
        X[ds]['zT']=CV['zT'][CV[ds+'_ind'],:].copy()
        X[ds]['zE']=CV['zE'][CV[ds+'_ind'],:].copy()
        X[ds]['orig_cluster']=O['cluster'][CV[ds+'_ind']].copy()
        X[ds]['cluster']=np.array(['rem']*X[ds]['zT'].shape[0],dtype=X[ds]['orig_cluster'].dtype)


    #Relabel
    for ds in ['train','val','test']:
        for label in this_classification:
            #Leaf nodes have an empty value in descendant_dict
            if len(descendant_dict[label])==0:
                X[ds]['cluster'][np.isin(X[ds]['orig_cluster'],label)] = label
            else:
                X[ds]['cluster'][np.isin(X[ds]['orig_cluster'],descendant_dict[label])] = label

    #Remove labels not in current classification
    for ds in ['train','val','test']:
        keep = X[ds]['cluster']!='rem'
        X[ds]['zT']=X[ds]['zT'][keep,:]
        X[ds]['zE']=X[ds]['zE'][keep,:]
        X[ds]['cluster']=X[ds]['cluster'][keep]
        X[ds]['orig_cluster']=X[ds]['orig_cluster'][keep]
    
    return X


def set_paths(exp_name='logistic_classifiers'):
    """Set data paths
    """

    from pathlib import Path   
    dir_pth = {}
    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'

    dir_pth['cvfolds'] = base_path + 'dat/result/TE_Patchseq_Bioarxiv/'
    dir_pth['result'] = dir_pth['cvfolds'] + exp_name + '/'

    Path(dir_pth['result']).mkdir(parents=True, exist_ok=True)
    return dir_pth


parser = argparse.ArgumentParser()
parser.add_argument("--cvfold",     default=0,                     type=int,   help='cv fold')
parser.add_argument("--alpha_T",    default=1.0,                   type=float, help='one of [1.0]')
parser.add_argument("--alpha_E",    default=0.5,                   type=float, help='one of [0.1,0.2,0.5,1.0]')
parser.add_argument("--lambda_TE",  default=1.0,                   type=float, help='one of [0.0,0.1,1.0,10.0]')
parser.add_argument("--root_node",  default='n88',                 type=str,   help='one of [n88,n60]')
parser.add_argument("--start_i",    default=0,                     type=int,   help='classification ids')
parser.add_argument("--stop_i",     default=100,                   type=int,   help='stop_i will be min of this and total number of classifications to be tested')
parser.add_argument("--embedding",  default='zE',                  type=str,   help='zE or zT')
parser.add_argument("--rand_seed",  default=0,                     type=int,   help='for repeatability/ measuring variability from initializing the logistic classifier')
parser.add_argument("--exp_name",   default='logistic_classifiers',type=str,   help='for repeatability/ measuring variability from initializing the logistic classifier')


def main(cvfold=0,
         alpha_T=1.0,
         alpha_E=0.5,
         lambda_TE=1.0,
         root_node='n88',
         start_i=0,
         stop_i=11000,
         embedding='zE',
         rand_seed=0,
         exp_name='logistic_classifiers'):

    alpha_M=alpha_E
    cvfold_fname='v1_aT_'+str(alpha_T)+\
                '_aE_'+str(alpha_E)+\
                '_aM_'+str(alpha_M)+\
                '_cs_'+str(lambda_TE)+\
                '_ld_3_bs_200_se_500_ne_1500_cv_'+str(cvfold)+\
                '_ri_0-summary'
    cvfold_fname=cvfold_fname.replace('.','-')+'.mat'
    dir_pth = set_paths(exp_name=exp_name)
    
    #Load pruned tree, embeddings, and cell type annotations
    with open(dir_pth['data']+"PS_v4_beta_0-4_matched_well-sampled_dend_RData_Tree_20181220_pruned_n88_n60_classifications.json") as f:
        all_classifications = json.load(f)        
    O = sio.loadmat(dir_pth['data']+'PS_v4_beta_0-4_matched_well-sampled.mat',squeeze_me=True)
    CV = sio.loadmat(dir_pth['cvfolds']+cvfold_fname,squeeze_me=True)
    htree_df = pd.read_csv(dir_pth['data']+'dend_RData_Tree_20181220_pruned.csv')
    htree = HTree(htree_df=htree_df)
    all_descendants = htree.get_all_descendants()
                
    result_fname = 'cv_classification_results_' + \
                    embedding + '-' + \
                    'aT_'+str(alpha_T)+'_' \
                    'aE_'+str(alpha_E)+'_' \
                    'aM_'+str(alpha_M)+'_' \
                    'csTE_'+str(lambda_TE) + \
                    '_randseed_'+str(rand_seed) + \
                    '_start_'+str(start_i) + \
                    '_stop_'+str(stop_i)
    result_fname = result_fname.replace('.','-')+'.csv'

    max_i = min(stop_i,len(all_classifications[root_node]))
    for i in range(start_i,max_i,1):
        print('Iter {:6d} in range {:6d} to {:6d}'.format(i,start_i,max_i))
        classification_id = root_node+'_'+str(i)
        this_classification = all_classifications[root_node][i]
        n_classes=len(this_classification)

        #Classifier only works for n_classes > 1 
        if n_classes>1: 
            X = relabel_restrict_inputs(CV=CV,O=O,this_classification=this_classification,descendant_dict=all_descendants)
            clf = LogisticRegression(penalty='none',
                                    random_state=rand_seed,
                                    solver='saga',
                                    max_iter=10000,
                                    multi_class='multinomial').fit(X['train'][embedding],X['train']['cluster'])
            
            result={}
            for ds in ['train','val','test']:
                pred_label = clf.predict(X[ds][embedding])
                result[ds+'_acc'] = np.sum(pred_label==X[ds]['cluster'])/X[ds]['cluster'].size
                result[ds+'_ari'] = adjusted_rand_score(X[ds]['cluster'], pred_label)
                result[ds+'_samples'] = pred_label.size

            result_list = [result['train_acc'], result['val_acc'], result['test_acc'],
                           result['train_ari'], result['val_ari'], result['test_ari'],
                           result['train_samples'], result['val_samples'], result['test_samples'],
                           cvfold, classification_id, n_classes]
                
            with open(dir_pth['result']+result_fname,'a') as f:
                writer = csv.writer(f)
                writer.writerows([result_list])
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
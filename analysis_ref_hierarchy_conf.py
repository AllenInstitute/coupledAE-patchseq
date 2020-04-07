import json
import pdb
import pickle
from copy import deepcopy
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns

from sklearn.metrics import adjusted_rand_score as ari
from timebudget import timebudget
from tqdm import tqdm
from pathlib import Path   
    
from analysis_tree_helpers import *


def relabel_ordered_classes_nonhierarchical(data_labels,ref_labels,ref_htree):
    """
    `data_labels` numpy array is updated based on `ref_labels` as per hierarchy in `ref_htree`.
    `data_labels` not appearing ref_labels or any child node of ref_labels are left unchanged.

    returns:
    new_data_labels
    new_data_label_order
    """
    new_data_labels = data_labels.copy()
    for label in ref_labels:
        labels_to_merge = ref_htree.get_descendants(label,leafonly=True)
        new_data_labels[np.isin(new_data_labels,labels_to_merge)]=label

    return new_data_labels


parser = argparse.ArgumentParser()
parser.add_argument("--cvfold", default=0, type=int, help="Fold id on which to perform ARI calculations")
def main(cvfold=0):


    curr_path = str(Path().absolute())
    if '/home/rohan' in curr_path:
        base_path = '/home/rohan/Remote-AI/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
    
    #Read partitions from file:
    partitions_path = base_path + 'dat/raw/patchseq-v4/'
    partitions_fname = 'PS_v5_beta_0-4_pc_scaled_ipxf_eqTE_n88_n60_classifications.json'
    with open(partitions_path+partitions_fname) as f:
        partitions = json.load(f)

    #Read modified tree from file:  
    tree_path = base_path + 'dat/raw/patchseq-v4/'
    tree_fname = 'PS_v5_beta_0-4_pc_scaled_ipxf_eqTE_well-sampled_inh_tree.json'
    tree_df = pd.read_csv(tree_path+tree_fname)
    htree = HTree(htree_df=tree_df)

    #Paths for original data files, representations, gmm fits.
    origdata_pth = base_path + 'dat/raw/patchseq-v4/PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat'
    representation_pth = base_path + 'dat/result/TE_aug_decoders/'
    gmm_pth = base_path + 'dat/result/TE_aug_decoders/gmm_fits_us/'
    
    #Load original data
    O = sio.loadmat(origdata_pth,squeeze_me=True)

    #Representation parameters
    alpha_T=1.0
    alpha_E=1.0
    lambda_TE=1.0
    gmm_n_components=30

    #Load all CV sets
    CVdict={}
    key_list = ['zE','zT','train_ind','val_ind','test_ind']

    cvfold_fname='v3_Edat_pcipfx_aT_'+str(alpha_T)+\
                '_aE_'+str(alpha_E)+\
                '_cs_'+str(lambda_TE)+\
                '_ld_3_bs_200_se_500_ne_1500_cv_'+str(cvfold)+\
                '_ri_0500_ft-summary'
    cvfold_fname=cvfold_fname.replace('.','-')+'.mat'
    X = sio.loadmat(representation_pth+cvfold_fname,squeeze_me=True)
    CVdict[cvfold] = {key:X[key] for key in key_list}
    del X

    fname = 'gmmfit_aT_{:.1f}_aE_{:.1f}_cs_{:.1f}_cv_{:d}_ld_3_ne_1500_fiton_zT_n_{:d}.pkl'.format(alpha_T,
                                                                                                alpha_E,
                                                                                                lambda_TE,
                                                                                                cvfold,
                                                                                                gmm_n_components)
    with open(gmm_pth+fname, 'rb') as fid:
        gmm = pickle.load(fid)

    #Label predictions using GMM fits:
    CVdict[cvfold]['ccT_lbl'] = gmm.predict(CVdict[cvfold]['zT'])
    CVdict[cvfold]['ccE_lbl'] = gmm.predict(CVdict[cvfold]['zE'])

    #Calculate ARI after merging based on the hierarchical tree
    ari_vals = np.empty(shape=(len(partitions),1))
    kept_cells = np.isin(O['cluster'],htree.child[htree.isleaf])

    fileid = 'v3_Edat_pcipfx_aT_'+str(alpha_T)+\
                    '_aE_'+str(alpha_E)+\
                    '_cs_'+str(lambda_TE)
    fileid = fileid.replace('.','-')
    save_pth = base_path + 'dat/result/TE_aug_decoders/ari_ref_partitions_cv_{}'.format(cvfold) + fileid+'.csv'

    for i in tqdm(range(len(partitions))):
        new_labels = relabel_ordered_classes_nonhierarchical(data_labels=O['cluster'],
                                                            ref_labels=partitions[i],
                                                            ref_htree=htree)

        ari_vals[i] = ari(new_labels[kept_cells],CVdict[cvfold]['ccT_lbl'][kept_cells])
        
        if (i+1)%1000==0:
            df = pd.DataFrame(ari_vals[:i],columns=['ARI'])
            df.to_csv(save_pth)

    df = pd.DataFrame(ari_vals,columns=['ARI'])
    df.to_csv(save_pth)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

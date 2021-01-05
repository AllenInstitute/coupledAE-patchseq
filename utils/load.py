import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io as sio
from refactor.utils.tree_helpers import get_merged_ordered_classes


def load_dataset(data_path='./refactor/data/proc/', min_sample_thr=10,):
    """Load input transcriptomic and electrophysiological profiles and label annotations as a dictionary.

    Args:
        data_path (str): Defaults to './refactor/data/proc/'.
        min_sample_thr (int): Defaults to 10.

    Returns:
        data(Dict)
    """

    data = sio.loadmat(data_path + 'PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat', squeeze_me=True)
    del_keys = [key for key in data.keys() if '__' in key]
    for key in del_keys:
        data.pop(key)

    with open(data_path + 'E_names.json') as f:
        ephys_names = json.load(f)
    data['E_pcipfx'] = np.concatenate([data['E_pc_scaled'], data['E_feature']], axis=1)
    data['pcipfx_names'] = np.concatenate([data['pc_name'],data['feature_name']])
    temp = [ephys_names[f] for f in data['pcipfx_names']]
    data['pcipfx_names'] = np.array(temp)
    
    #Get t-types in order as per reference taxonomy
    n_required_classes = np.unique(data['cluster']).size
    _, t_types = get_merged_ordered_classes(data_labels=data['cluster'].copy(),
                                            htree_file=data_path+'dend_RData_Tree_20181220.csv',
                                            n_required_classes=n_required_classes,
                                            verbose=False)
    data['unique_sorted_t_types']=np.array(t_types)

    #well-sampled t-types and helpers:
    t_types_well_sampled = []
    for t in t_types:
        if np.sum(data['cluster']==t)>min_sample_thr:
            t_types_well_sampled.append(t)
    data['well_sampled_sorted_t_types'] = np.array(t_types_well_sampled)
    data['well_sampled_bool'] = np.isin(data['cluster'],data['well_sampled_sorted_t_types'])
    data['well_sampled_ind'] = np.flatnonzero(data['well_sampled_bool'])

    #Process E data and mask, standardize names.
    data['XT'] = data['T_dat']
    data['XE'] = np.concatenate([data['E_pc_scaled'],data['E_feature']],axis = 1)
    data['maskE'] = np.ones_like(data['XE'])
    data['maskE'][np.isnan(data['XE'])]=0.0
    data['XE'][np.isnan(data['XE'])]=0.0
    return data


def load_summary_files(data_type='NM_cc',key_list = ['XrE','XrT','zE','zT','train_ind','val_ind','test_ind'],**kwargs):
    """Loads saved output of autoencoder runs for specified experiment. 
    Output is a dict with different runs as keys. 

    Args:
        data_type (str, optional): One of 'NM' ,'NM_cc', 'CS'. Defaults to 'cc'.
        key_list (list, optional): [description]. Defaults to ['XrE','XrT','zE','zT','train_ind','val_ind','test_ind'].

    Returns:
        CVdict: dictionary with each run as a key
    """
    
    O = load_dataset()    
    CVdict={}

    #'NM_cc' has data for 21 repeats on the same train/test split
    if data_type=='NM_cc':
        alpha_T = kwargs.get('alpha_T',1.0)
        alpha_E = kwargs.get('alpha_E',1.0)
        lambda_TE = kwargs.get('lambda_TE',1.0)
        aug = kwargs.get('aug',1)
        fold_list = kwargs.get('fold_list',list(range(21)))
        data_path = kwargs.get('data_path','/home/rohan/Remote-AI/dat/result/TE_NM_cc/')
        gmm_path = kwargs.get('gmm_path','/home/rohan/Remote-AI/dat/result/TE_NM_cc/gmm_model_select_cv_0/')
        best_n_components = kwargs.get('best_n_components',33)
        load_gmm = kwargs.get('load_gmm',True)
        cv=0
        
        for fold in fold_list:
            cvfold_fname = (f'NM_Edat_pcipfx_aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}' +
                            f'_ad_{aug:d}_ld_3_bs_200_se_500_ne_1500_cv_{cv:d}_ri_{fold}_500_ft-summary').replace('.', '-') + '.mat'
            
            if Path(data_path+cvfold_fname).is_file():
                X = sio.loadmat(data_path+cvfold_fname,squeeze_me=True)
                CVdict[fold] = {key:X[key] for key in key_list}
                CVdict[fold]['well_sampled_test_ind'] = X['test_ind'][np.isin(X['test_ind'],O['well_sampled_ind'])]
                CVdict[fold]['well_sampled_train_ind'] = X['train_ind'][np.isin(X['train_ind'],O['well_sampled_ind'])]

                if load_gmm: #Assign labels to unsupervised clusters
                    gmm_fname = (f'gmmfit_restricted_perc_100-0_aT_{alpha_T:.1f}_aE_{alpha_E:.1f}_cs_{lambda_TE:.1f}_'+
                                 f'ad_1_cv_0_ri_{fold:d}_ld_3_ne_1500_fiton_zT_n_{best_n_components:d}').replace('.','-')+'.pkl'
                    with open(gmm_path+gmm_fname, 'rb') as fid:
                        gmm = pickle.load(fid)
                        X['ccT_lbl'] = gmm.predict(X['zT'])
                        X['ccE_lbl'] = gmm.predict(X['zE'])
                        t_lbl,e_lbl = relabel_gmm_clusters(X=X,O=O.copy(),best_n_components=best_n_components)
                        CVdict[fold]['ccT_lbl_matched'],CVdict[fold]['ccE_lbl_matched']=t_lbl,e_lbl
                del X

            else: print(data_path+cvfold_fname+' not found')
                
    #'NM' has data for 44 cross validation splits. 
    elif data_type=='NM':
        alpha_T = kwargs.get('alpha_T',1.0)
        alpha_E = kwargs.get('alpha_E',1.0)
        lambda_TE = kwargs.get('lambda_TE',1.0)
        aug = kwargs.get('aug',1)
        fold_list = kwargs.get('fold_list',list(range(44)))
        data_path = kwargs.get('data_path','/home/Rohan/Remote-AI/dat/result/TE_NM/')
        print(f'Loading aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}_ad_{aug:d}')

        ri=0
        for fold in fold_list:
            cvfold_fname = (f'NM_Edat_pcipfx_aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}' +
                            f'_ad_{aug:d}_ld_3_bs_200_se_500_ne_1500_cv_{fold:d}_ri_{ri}_500_ft-summary').replace('.', '-') + '.mat'
            
            if Path(data_path+cvfold_fname).is_file():
                X = sio.loadmat(data_path+cvfold_fname,squeeze_me=True)
                CVdict[fold] = {key:X[key] for key in key_list}
                del X
            else: print(cvfold_fname,'not found')

    return CVdict


def relabel_gmm_clusters(X:Dict, datadict: Dict, best_n_components:int):
    """Given a collection of T and E clusters, finds the best match with  
    reference taxonomy labels (ordered nbased on the hierarchical tree) and 
    relabels the clusters with that order.

    Args:
        X (Dict): Contains predictions of the coupled autoencoder network
        datadict (Dict): various dataset fields
        best_n_components (int): [description]

    Returns:
        ccT_lbl_matched, ccE_lbl_matched: Labels for the T and E samples that are an optimal 
        match w.r.t the transcriptomic taxonomy labels.
    """
    from refactor.utils.compute import contingency
    from scipy.optimize import linear_sum_assignment

    for key in ['train_ind', 'ccT_lbl', 'ccE_lbl']:
        assert key in X.keys(), 'Input does not contain required fields'

    for key in ['unique_sorted_t_types', 'cluster']:
        assert key in datadict.keys(), 'Input does not contain required fields'

    #Calculate contingency matrix based on label assignments
    C = contingency(a=datadict['cluster'][X['train_ind']],
                    b=X['ccT_lbl'][X['train_ind']],
                    unique_a=datadict['unique_sorted_t_types'].copy(),
                    unique_b=np.arange(best_n_components))

    #Hungarian algorithm assignments:
    row_ind, col_ind = linear_sum_assignment(-C)
    C_ordered = C[:, col_ind]
    order_y = np.arange(0, best_n_components)[col_ind]

    ccT_lbl_matched = X['ccT_lbl'].copy()
    ccE_lbl_matched = X['ccE_lbl'].copy()

    for i in range(best_n_components):
        ind = X['ccT_lbl'] == order_y[i]
        ccT_lbl_matched[ind] = i

        ind = X['ccE_lbl'] == order_y[i]
        ccE_lbl_matched[ind] = i

    return ccT_lbl_matched, ccE_lbl_matched


def taxonomy_assignments(initial_labels, datadict: Dict, n_required_classes: int, merge_on='well_sampled'):
    """Relabels the the input labels according to a lower resolution of the reference taxonomy. 

    Args:
        initial_labels: initial labels. np.array or List.
        datadict (Dict): dictionary with fields `well_sampled_sorted_t_types` or `unique_sorted_t_types`
        n_required_classes (int): assumed to be less than or equal to number of classes in the taxonomy
        merge_on (str, optional): Subset of classes of the taxonomy. Defaults to 'well_sampled'.

    Returns:
        updated_labels (np.array): new labels, same size as initial labels
        n_remain_classes (int)
    """

    reference_labels = None
    if merge_on == 'well_sampled':
        assert 'well_sampled_sorted_t_types' in datadict.keys(), 'Required key missing'
        reference_labels = datadict['well_sampled_sorted_t_types'].copy()
    elif merge_on == 'all':
        assert 'unique_sorted_t_types' in datadict.keys(), 'Required key missing'
        reference_labels = datadict['unique_sorted_t_types'].copy()
    assert reference_labels is not None, "reference_labels not set"

    new_reference_labels, _ = get_merged_ordered_classes(data_labels=reference_labels.copy(),
                                                         htree_file='./refactor/data/proc/dend_RData_Tree_20181220.csv',
                                                         n_required_classes=n_required_classes,
                                                         verbose=False)

    n_remain_classes = np.unique(new_reference_labels).size
    updated_labels = np.array(initial_labels.copy())
    updated_labels[~np.isin(updated_labels, reference_labels)] = 'not_well_sampled'
    for (orig, new) in zip(reference_labels, new_reference_labels):
        updated_labels[updated_labels == orig] = new

    return updated_labels, n_remain_classes

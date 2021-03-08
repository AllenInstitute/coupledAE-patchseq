import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def reorder_ps_TE(ps_T_dat,ps_T_ann,ps_E_dat):
    '''Concatenates data with paired T and E cells first, and exclusive cells later
    \nOutputs: `ps_Tcat_dat`, `ps_Tcat_ann`, `ps_Ecat_dat`, `ispairedT`, `ispairedE`'''
    #Transcriptomic exclusive:
    Tonly_bool = ~np.isin(ps_T_ann['spec_id_label'].values.astype(int),ps_E_dat['spec_id_label'].values)
    ps_Tonly_ann = ps_T_ann.iloc[Tonly_bool].copy()
    ps_Tonly_dat = ps_T_dat.iloc[Tonly_bool].copy()

    #Transcriptomic with match in Ephys:
    TinE_bool = np.isin(ps_T_ann['spec_id_label'].values.astype(int),ps_E_dat['spec_id_label'].values)
    ps_TinE_ann = ps_T_ann.iloc[TinE_bool].copy()
    ps_TinE_dat = ps_T_dat.iloc[TinE_bool].copy()

    print('-----------------------------------------------')
    print("{} exclusive, {} matched, total {} in T".format(
        np.sum(Tonly_bool), np.sum(TinE_bool), ps_T_dat.shape[0]))

    #Ephys exclusive:
    Eonly_bool = ~np.isin(ps_E_dat['spec_id_label'].values,ps_T_ann['spec_id_label'].values.astype(int))
    ps_Eonly_dat = ps_E_dat.iloc[Eonly_bool].copy()
    
    #Ephys with match in Transcriptomic:
    EinT_bool = np.isin(ps_E_dat['spec_id_label'].values,ps_T_ann['spec_id_label'].values.astype(int))
    ps_EinT_dat = ps_E_dat.iloc[EinT_bool].copy()

    print('-----------------------------------------------')
    print("{} exclusive, {} matched, total {} in E".format(
        np.sum(Eonly_bool), np.sum(EinT_bool), ps_E_dat.shape[0]))

    #Enforce order of matched data to be same for Transcriptomics and Ephys
    #T annotations and E features are matched through spec_id_label
    #T data and annotations are aligned through indexing
    ps_EinT_dat.sort_values('spec_id_label',inplace = True)          # Contains ephys features
    ps_TinE_ann.sort_values('spec_id_label',inplace = True)          # Contains transcriptome annotations
    ps_TinE_dat = ps_TinE_dat.reindex(ps_TinE_ann.index,copy=True)   # Contains gene expression count data

    #Concatenate transcriptomic data
    ps_Tcat_dat = pd.concat([ps_TinE_dat, ps_Tonly_dat], ignore_index=True)
    ps_Tcat_ann = pd.concat([ps_TinE_ann, ps_Tonly_ann], ignore_index=True)
    ispairedT = np.concatenate((np.ones((ps_TinE_dat.shape[0],)),np.zeros((ps_Tonly_dat.shape[0],))),axis=0)

    #Concatenate Ephys data
    ps_Ecat_dat = pd.concat([ps_EinT_dat, ps_Eonly_dat], ignore_index=True)
    ispairedE = np.concatenate((np.ones((ps_EinT_dat.shape[0],)),np.zeros((ps_Eonly_dat.shape[0],))),axis=0)

    return ps_Tcat_dat, ps_Tcat_ann, ps_Ecat_dat, ispairedT, ispairedE


def extract_arrays(ps_Tcat_dat, ps_Tcat_ann, ispairedT, ps_Ecat_dat, ispairedE, keep_gene_id):
    """Crops the data to only genes of interest.
    Output: `matdict` dictionary to save in .mat format
    """
    ps_gene_id = np.array(ps_Tcat_dat.columns[1:],dtype=np.object)
    keep_gene_index = np.where(np.isin(ps_gene_id,keep_gene_id))[0]
    
    matdict={}
    #0th column is sample_id in the ps_Tcat_dat array
    matdict['T_dat'] = np.log1p(ps_Tcat_dat.values[:, 1:].astype(np.float32))[:, keep_gene_index]
    matdict['T_spec_id_label'] = ps_Tcat_ann.loc[:,'spec_id_label'].values.astype(int)
    matdict['T_ispaired'] = ispairedT

    #Transcriptomic annotations:
    matdict['gene_id'] = ps_Tcat_dat.columns[1:].values[keep_gene_index]
    matdict['cluster'] = ps_Tcat_ann.loc[:, 'cluster_label'].values
    matdict['clusterID'] = ps_Tcat_ann.loc[:, 'cluster_id'].values.astype(int)
    matdict['cluster_color'] = ps_Tcat_ann.loc[:, 'cluster_color'].values
    matdict['sample_id'] = ps_Tcat_ann.loc[:, 'sample_id'].values
    matdict['map_conf'] = ps_Tcat_ann.loc[:, 'map_conf'].values

    matdict['E_dat'] = ps_Ecat_dat.values[:, 1:].astype(np.float32)
    matdict['E_spec_id_label'] = ps_Ecat_dat.loc[:,'spec_id_label'].values.astype(int)
    matdict['E_ispaired'] = ispairedE
    return matdict


def TE_get_splits(matdict):
    """Creates 1 test set, and from the remaining cells creates 9 other sets for 9-fold cross validation
    Data in the test and validation sets is stratified based on T cluster labels.
    
    Returns:
        cvset -- list with 9 cross-validation folds. Train indices: cvset[0]['train'] and Validation indices: cvset[0]['val']
        testset -- indices for test cells
    Arguments:
        matdict -- dataset dictionary
    """
    assert np.array_equal(matdict['T_spec_id_label'], matdict['E_spec_id_label']), 'Sample order is not the same across datasets'
    X = matdict['T_dat']
    y = matdict['cluster']

    #Split data into 10 folds first deterministically.
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    ind_dict = [{'train': train_ind, 'val': val_ind}
                for train_ind, val_ind in skf.split(X, y)]

    #Define the test set
    testset = ind_dict[0]['val']

    #Remove test cells from remaining folds (expected to overlap only with the training cells).
    cvset = []
    for i in range(1, len(ind_dict), 1):
        ind_dict[i]['train'] = np.setdiff1d(ind_dict[i]['train'], testset)
        ind_dict[i]['val'] = np.setdiff1d(ind_dict[i]['val'], testset)
        cvset.append(ind_dict[i])

    return cvset, testset


def TE_get_splits_45(matdict, cvfold):
    """Creates a test set with ~10% of the samples. Remaining cells are used for 45-fold cross validation
    Test and validation sets are stratified based on T cluster labels.
    
    Returns:
        cvset -- list with 45 cross-validation folds. Train indices: cvset[0]['train'] and Validation indices: cvset[0]['val']
        testset -- indices for test cells
    Arguments:
        matdict -- dataset dictionary
    """

    skf = StratifiedKFold(n_splits=50, random_state=0, shuffle=True)
    ind_dict = [{'train': train_ind, 'val': val_ind} for train_ind, val_ind in skf.split(X=np.zeros(shape=matdict['cluster'].shape), y=matdict['cluster'])]

    #Pool a fraction of the stratified folds to define the test set
    test_ind = []
    for i in range(45, 50, 1):
        test_ind.append(ind_dict[i]['val'])
    test_ind = np.concatenate(test_ind)

    #Ensure test set cells do not appear in any of the training sets.
    cvset = []
    for i in range(1, len(ind_dict), 1):
        ind_dict[i]['train'] = np.setdiff1d(ind_dict[i]['train'], test_ind)
        ind_dict[i]['val'] = np.setdiff1d(ind_dict[i]['val'], test_ind)
        cvset.append(ind_dict[i])

    train_ind = cvset[cvfold]['train']
    val_ind = cvset[cvfold]['val']
    return train_ind, val_ind, test_ind


def TE_get_splits_5(matdict, cvfold):
    """Creates 80-20 split stratified based on T cluster labels. val_ind is same as test_ind
    
    Returns:
        cvset -- list with 5 cross-validation folds. 
        testset -- indices for test cells (same as validation set)
    Arguments:
        matdict -- dataset dictionary
    """

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    ind_dict = [{'train': train_ind, 'val': val_ind} for train_ind, val_ind in skf.split(X=np.zeros(shape=matdict['cluster'].shape), y=matdict['cluster'])]
    train_ind = ind_dict[cvfold]['train']
    val_ind = ind_dict[cvfold]['val']
    test_ind = ind_dict[cvfold]['val']
    return train_ind, val_ind, test_ind

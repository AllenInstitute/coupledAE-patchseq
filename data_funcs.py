import numpy as np
import scipy.io as sio
import feather
import pandas as pd
import timeit
import pdb

def create_color_ref():
    '''Used to generate `cluster_colors` from the FACS reference dataset'''
    D = sio.loadmat('/Users/fruity/Dropbox/AllenInstitute/CellTypes/dat/raw/Mouse-V1-ALM-20180520_cpmtop10k_cpm.mat',squeeze_me=True)
    ctype_list = []
    col_list = []
    for ctype,col in list(set(zip(D['cluster'],D['cluster_color']))):
        ctype_list.append(ctype)
        col_list.append(col)
    D = pd.DataFrame({'celltype':ctype_list,'cluster_color':col_list})
    D = D.loc[D['celltype'].isin(np.unique(ps_Tcat_ann['topLeaf_label'].values))]
    D = D.reset_index(drop=True)
    D.to_csv(base_path + 'type_color_reference.csv',index=False)
    return

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
    '''Crops the data to only genes of interest.
    Output: `matdict` dictionary to save in .mat format
    '''
    ps_gene_id = np.array(ps_Tcat_dat.columns[1:],dtype=np.object)
    keep_gene_index = np.where(np.isin(ps_gene_id,keep_gene_id))[0]
    
    matdict={}
    #0th column is sample_id in the ps_Tcat_dat array
    matdict['T_dat']           = np.log1p(ps_Tcat_dat.values[:,1:].astype(np.float32))[:,keep_gene_index]
    matdict['T_spec_id_label'] = ps_Tcat_ann.loc[:,'spec_id_label'].values.astype(int)
    matdict['T_ispaired']      = ispairedT

    #Transcriptomic annotations:
    matdict['gene_id']         = ps_Tcat_dat.columns[1:].values[keep_gene_index]
    matdict['cluster']         = ps_Tcat_ann.loc[:,'cluster_label'].values
    matdict['clusterID']       = ps_Tcat_ann.loc[:,'cluster_id'].values.astype(int)
    matdict['cluster_color']   = ps_Tcat_ann.loc[:,'cluster_color'].values
    matdict['sample_id']       = ps_Tcat_ann.loc[:,'sample_id'].values

    matdict['E_dat']           = ps_Ecat_dat.values[:,1:].astype(np.float32)    
    matdict['E_spec_id_label'] = ps_Ecat_dat.loc[:,'spec_id_label'].values.astype(int)
    matdict['E_ispaired']      = ispairedE
    return matdict


def shuffle_dataset_mat(matdict):
    '''Data will be deterministically shuffled'''
    np.random.seed(seed=10)

    #Common shuffled indices for paired data
    shuffle_TE = np.where(matdict['T_ispaired']==1)[0]
    np.random.shuffle(shuffle_TE)

    #Shuffled indices for T exclusive data
    shuffle_Tonly = np.where(matdict['T_ispaired']==0)[0]
    np.random.shuffle(shuffle_Tonly)

    #Shuffled indices for E exclusive data
    shuffle_Eonly =  np.where(matdict['E_ispaired']==0)[0]
    np.random.shuffle(shuffle_Eonly)

    #Shuffle arrays:
    shuffle_T = np.concatenate((shuffle_TE,shuffle_Tonly),axis = 0)
    shuffle_E = np.concatenate((shuffle_TE,shuffle_Eonly),axis = 0)
    shuffle_M = shuffle_E.copy()

    matdict['T_dat']           = matdict['T_dat'][shuffle_T,:]
    matdict['T_spec_id_label'] = matdict['T_spec_id_label'][shuffle_E]
    matdict['T_ispaired']      = matdict['T_ispaired'][shuffle_T]
    matdict['cluster']         = matdict['cluster'][shuffle_T]       
    matdict['clusterID']       = matdict['clusterID'][shuffle_T]
    matdict['cluster_color']   = matdict['cluster_color'][shuffle_T]
    matdict['sample_id']       = matdict['sample_id'][shuffle_T] 

    matdict['E_dat']           = matdict['E_dat'][shuffle_E,:]     
    matdict['E_spec_id_label'] = matdict['E_spec_id_label'][shuffle_E]
    matdict['E_ispaired']      = matdict['E_ispaired'][shuffle_E]  

    matdict['M_dat']           = matdict['M_dat'][shuffle_M,:]
    return matdict

def TEM_dataset(matdict):
    '''Returns dictionary in which only data common to all three modalities is retained.
    Samples in the same row index are the same across modalities for paired recordings.'''

    print('{:d} M datapoints have nans'.format(np.sum(np.isnan(matdict['M_dat']))))
    keep_cells = np.intersect1d(np.flatnonzero(matdict['T_ispaired']==1), np.flatnonzero(matdict['E_ispaired']==1))
    keep_cells = np.intersect1d(keep_cells, np.flatnonzero(~np.isnan(matdict['M_dat'])))
    
    TEM_matdict={}
    TEM_matdict['T_dat']           = matdict['T_dat'][keep_cells,:]
    TEM_matdict['T_spec_id_label'] = matdict['T_spec_id_label'][keep_cells]
    TEM_matdict['T_ispaired']      = matdict['T_ispaired'][keep_cells]
    TEM_matdict['cluster']         = matdict['cluster'][keep_cells]       
    TEM_matdict['clusterID']       = matdict['clusterID'][keep_cells]
    TEM_matdict['cluster_color']   = matdict['cluster_color'][keep_cells]
    TEM_matdict['sample_id']       = matdict['sample_id'][keep_cells] 

    TEM_matdict['E_dat']           = matdict['E_dat'][keep_cells,:]     
    TEM_matdict['E_spec_id_label'] = matdict['E_spec_id_label'][keep_cells]
    TEM_matdict['E_ispaired']      = matdict['E_ispaired'][keep_cells]  

    TEM_matdict['M_dat']           = matdict['M_dat'][keep_cells]

    assert np.array_equal(TEM_matdict['E_spec_id_label'],TEM_matdict['E_spec_id_label']),'Sample order is not the same across datasets'
    assert np.size(TEM_matdict['M_dat'])==np.shape(TEM_matdict['T_dat'])[0],'Dataset sizes are mismatched'
    assert np.size(TEM_matdict['M_dat'])==np.shape(TEM_matdict['E_dat'])[0],'Dataset sizes are mismatched'
    print('{:d} remaining samples'.format(np.size(TEM_matdict['M_dat'])))
    print('{:d} remaining T-types '.format(np.size(np.unique(TEM_matdict['cluster']))))
    return TEM_matdict
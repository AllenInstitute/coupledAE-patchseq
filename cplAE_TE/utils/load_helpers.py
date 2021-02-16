import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io as sio
import toml
from cplAE_TE.utils.tree_helpers import HTree, get_merged_ordered_classes, simplify_tree


def get_paths(warn=True, write_toml=False):
    """Get paths for all data used in the analysis. 
    
    Args: 
    warn (bool): Warns when files or directories are missing. 
    write_toml (bool): Writes out a .toml file in the notebooks folder with paths as strings.

    Returns:
    path : a dictionary with many different paths.  
    """
    path = {}
    path['package'] = Path(__file__).parent.parent.parent.absolute()

    # Input data
    path['proc_dataset'] = path['package'] / "data/proc/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat"
    path['proc_E_names'] = path['package'] / "data/proc/E_names.json"
    path['htree'] = path['package'] / "data/proc/dend_RData_Tree_20181220.csv"
    #path['htree_pruned'] = path['package'] / "data/proc/dend_RData_Tree_20181220_pruned.csv"
    for f in ['proc_dataset', 'proc_E_names', 'htree']:
        if (not(path[f].is_file()) and warn):
            print(f'File not found: {path[f]}')

    # Paths to data from different experiments
    remote_path = Path("/home/rohan/Remote-AI/dat/")
    path['exp_hparam'] = remote_path / "result/TE_NM/"
    path['exp_hparam_log'] = remote_path / "result/TE_NM/logs/"
    path['exp_kfold'] = remote_path / "result/TE_NM/"
    path['exp_repeat_init'] = remote_path / "result/TE_NM_cc/"
    path['exp_repeat_init_gmm'] = remote_path / "result/TE_NM_cc/gmm_model_select_cv_0/"
    for f in ['exp_hparam', 'exp_hparam_log', 'exp_kfold', 'exp_repeat_init', 'exp_repeat_init_gmm']:
        if (not(path[f].is_dir()) and warn):
            print(f'Directory not found: {path[f]}')

    if write_toml:
        with open(path['package'] / "notebooks/config_paths.toml",'w') as f:
            str_path = path.copy()
            for key in str_path.keys():str_path[key] = str(str_path[key])
            toml.dump(str_path,f)
    return path


def load_dataset(min_sample_thr=10):
    """Load input transcriptomic and electrophysiological profiles and label annotations as a dictionary.

    Args:
        min_sample_thr (int): Defaults to 10.

    Returns:
        data(Dict)
    """
    path = get_paths(warn=False,write_toml=False)
    data = sio.loadmat(path['proc_dataset'], squeeze_me=True)
    del_keys = [key for key in data.keys() if '__' in key]
    for key in del_keys:
        data.pop(key)

    with open(path['proc_E_names']) as f:
        ephys_names = json.load(f)
    data['E_pcipfx'] = np.concatenate([data['E_pc_scaled'], data['E_feature']], axis=1)
    data['pcipfx_names'] = np.concatenate([data['pc_name'],data['feature_name']])
    temp = [ephys_names[f] for f in data['pcipfx_names']]
    data['pcipfx_names'] = np.array(temp)
    
    #Get t-types in order as per reference taxonomy
    n_required_classes = np.unique(data['cluster']).size
    _, t_types = get_merged_ordered_classes(data_labels=data['cluster'].copy(),
                                            htree_file=path['htree'],
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


def load_htree_well_sampled(min_sample_thr=10, simplify=True):
    """Loads heirarchical taxonomy for subset of well-sampled inhibitory cell types (leaf nodes).

    Args:
        min_sample_thr (int, optional): n > min_sample_thr qualifies as well-sampled. Defaults to 10.
        simplify (bool, optional): Removes intermediate nodes when only one child node exists. Defaults to True.

    Returns:
        htree: HTree object
    """

    #Get inhibitory tree
    print('Only searching through inhibitory taxonomy (n59)')
    path = get_paths(warn=False, write_toml=False)
    htree = HTree(htree_file=path['htree'])
    subtree = htree.get_subtree(node='n59')

    #Get list of well-sampled cell types
    dataset = load_dataset(min_sample_thr=min_sample_thr)
    kept_classes = dataset['well_sampled_sorted_t_types'].tolist()
    print(f'{len(kept_classes)} cell types retained, with at least {min_sample_thr} samples in the dataset')

    #Get tree with kept_classes:
    kept_tree_nodes = []
    for node in kept_classes:
        kept_tree_nodes.extend(subtree.get_ancestors(node))
        kept_tree_nodes.extend([node])

    kept_htree_df = subtree.obj2df()
    kept_htree_df = kept_htree_df[kept_htree_df['child'].isin(kept_tree_nodes)]
    kept_htree = HTree(htree_df=kept_htree_df)

    #Simplify layout and plot
    if simplify:
        htree, _ = simplify_tree(kept_htree, skip_nodes=None, verbose=False)
        htree.update_layout()
    else:
        htree = kept_htree
    return htree


def load_summary_files(data_type='NM_cc', key_list=['XrE', 'XrT', 'zE', 'zT', 'train_ind', 'val_ind', 'test_ind'], **kwargs):
    """Loads saved output of autoencoder runs for specified experiment.
    Output is a dict with different runs as keys.

    Args:
        data_type (str, optional): One of 'NM' ,'NM_cc', 'CS'.
        key_list (list, optional): [description]. Defaults to ['XrE','XrT','zE','zT','train_ind','val_ind','test_ind'].

    Returns:
        CVdict: dictionary with each run as a key
    """
    O = load_dataset()
    path = get_paths()
    CVdict = {}

    #'NM_cc' has data for 21 repeats on the same train/test split
    if data_type == 'NM_cc':
        alpha_T = kwargs.get('alpha_T', 1.0)
        alpha_E = kwargs.get('alpha_E', 1.0)
        lambda_TE = kwargs.get('lambda_TE', 1.0)
        aug = kwargs.get('aug', 1)
        fold_list = kwargs.get('fold_list', list(range(21)))
        best_n_components = kwargs.get('best_n_components', 33)
        load_gmm = kwargs.get('load_gmm', True)
        cv = 0
        
         
        for fold in fold_list:
            cvfold_fname = (f'NM_Edat_pcipfx_aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}' +
                            f'_ad_{aug:d}_ld_3_bs_200_se_500_ne_1500_cv_{cv:d}_ri_{fold}_500_ft-summary').replace('.', '-') + '.mat'

            if (path['exp_repeat_init'] / cvfold_fname).is_file():
                X = sio.loadmat(path['exp_repeat_init'] / cvfold_fname, squeeze_me=True)
                CVdict[fold] = {key: X[key] for key in key_list}
                CVdict[fold]['well_sampled_test_ind'] = X['test_ind'][np.isin(X['test_ind'], O['well_sampled_ind'])]
                CVdict[fold]['well_sampled_train_ind'] = X['train_ind'][np.isin(X['train_ind'], O['well_sampled_ind'])]

                if load_gmm:  # Assign labels to unsupervised clusters.
                    #gmm.predict might fail if scikit-learn version is not the same (we used 0.22.2).
                    gmm_fname = (f'gmmfit_restricted_perc_100-0_aT_{alpha_T:.1f}_aE_{alpha_E:.1f}_cs_{lambda_TE:.1f}_' +
                                 f'ad_1_cv_0_ri_{fold:d}_ld_3_ne_1500_fiton_zT_n_{best_n_components:d}').replace('.', '-')+'.pkl'
                    with open(path['exp_repeat_init_gmm'] / gmm_fname, 'rb') as fid:
                        gmm = pickle.load(fid)
                        X['ccT_lbl'] = gmm.predict(X['zT'])
                        X['ccE_lbl'] = gmm.predict(X['zE'])
                        t_lbl, e_lbl = relabel_gmm_clusters(
                            X=X, datadict=O.copy(), best_n_components=best_n_components)
                        CVdict[fold]['ccT_lbl_matched'], CVdict[fold]['ccE_lbl_matched'] = t_lbl, e_lbl
                del X

            else:
                print(path+cvfold_fname+' not found')

    #'NM' has data for 44 cross validation splits. 
    elif data_type == 'NM':
        alpha_T = kwargs.get('alpha_T', 1.0)
        alpha_E = kwargs.get('alpha_E', 1.0)
        lambda_TE = kwargs.get('lambda_TE', 1.0)
        latent_dim = kwargs.get('latent_dim', 1.0)
        aug = kwargs.get('aug', 1)
        fold_list = kwargs.get('fold_list', list(range(44)))
        fstr = f'Loading aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}_ad_{aug:d}'
        print(fstr.replace('.','-'))

        ri = 0
        for fold in fold_list:
            cvfold_fname = (f'NM_Edat_pcipfx_aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}' +
                            f'_ad_{aug:d}_ld_{latent_dim:d}_bs_200_se_500_ne_1500_cv_{fold:d}_ri_{ri}_500_ft-summary').replace('.', '-') + '.mat'

            if (path['exp_kfold'] / cvfold_fname).is_file():
                X = sio.loadmat(path['exp_kfold'] / cvfold_fname, squeeze_me=True)
                CVdict[fold] = {key: X[key] for key in key_list}
                del X
            else:
                print(cvfold_fname, 'not found')

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
    from cplAE_TE.utils.compute import contingency
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

    path = get_paths(warn=False,write_toml=False)
    new_reference_labels, _ = get_merged_ordered_classes(data_labels=reference_labels.copy(),
                                                         htree_file=str(path['htree']),
                                                         n_required_classes=n_required_classes,
                                                         verbose=False)

    n_remain_classes = np.unique(new_reference_labels).size
    updated_labels = np.array(initial_labels.copy())
    updated_labels[~np.isin(updated_labels, reference_labels)] = 'not_well_sampled'
    for (orig, new) in zip(reference_labels, new_reference_labels):
        updated_labels[updated_labels == orig] = new

    return updated_labels, n_remain_classes




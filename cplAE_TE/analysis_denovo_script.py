import csv
import json
import argparse
import sys
import pdb
import pickle
import argparse

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import mixture
from timebudget import timebudget

def set_paths(representation_pth='TE_CS',exp_name='denovo_clustering'):
    """Set data paths
    """

    from pathlib import Path   
    dir_pth = {}
    curr_path = str(Path().absolute())
    base_path=None
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
    elif '/home/rohan' in curr_path:
        #base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
        base_path = '/home/rohan/Remote-AI/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
        
    dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'
    dir_pth['cvfolds'] = base_path + 'dat/result/'+representation_pth+'/'
    dir_pth['result'] = dir_pth['cvfolds'] + exp_name + '/'

    Path(dir_pth['result']).mkdir(parents=True, exist_ok=True)
    return dir_pth

parser = argparse.ArgumentParser()

parser.add_argument("--representation_pth",default='TE_CS',              type=str,    help='Directory to load representations from')
parser.add_argument("--exp_name",          default='gmm_model_select',   type=str,    help='Result folder')
parser.add_argument("--cvfold",            default=0,                    type=int,    help='CV set in [0,...,44]')
parser.add_argument("--modeltype",         default='pccca',              type=str,    help='pccca or cplae')
parser.add_argument("--min_component",     default=10,                   type=int,    help='min GMM components')
parser.add_argument("--max_component",     default=None,                 type=int,    help='max GMM components')
parser.add_argument("--perc",              default=100,                  type=float,  help='percent of best reconstructed data to keep')


def main(representation_pth='TE_CS',
         exp_name='gmm_model_select',
         cvfold=0,
         modeltype='pccca',
         min_component=10,
         max_component=None,
         perc=97):
    
    #GMM parameters
    if max_component is None:
        max_component = min_component+1
        
    n_components_range = np.arange(min_component,max_component)
    n_init = 50
    max_iter = int(1e4)
    tol = 1e-6

    #Representations
    fiton=['zT']
    fname=None
    cvfold_fname=None

    if modeltype=='cplae':
        alpha_T=1.0
        alpha_E=1.0
        lambda_TE=1.0
        augmented_decoders=1
        latent_dim = 3

        fname = (f'gmmfit_'+ \
                f'perc_{perc:0.1f}_ld_{latent_dim}_fiton_{"".join(fiton)}_aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}' + \
                f'_ad_{augmented_decoders}_cv_{cvfold:d}').replace('.','-')
        cvfold_fname = (f'CS_Edat_pcipfx_aT_{alpha_T:0.1f}_aE_{alpha_E:0.1f}_cs_{lambda_TE:0.1f}_ad_{augmented_decoders:d}_' + \
                        f'ld_{latent_dim:d}_bs_200_se_500_ne_1500_cv_{cvfold:d}_ri_0_best_loss-summary').replace('.','-')+'.mat'

    elif modeltype=='pccca':
        pca_dim=20
        cca_dim=3

        fname = (f'gmmfit_'+ \
                f'perc_{perc:0.1f}_pca_{pca_dim:d}_cca_{cca_dim:d}_fiton_{"".join(fiton)}_' + \
                f'cv_{cvfold:d}').replace('.','-')
        cvfold_fname = f'PCCCA_Edat_pcipfx_pcT_{pca_dim:d}_pcE_{pca_dim:d}_cca_{cca_dim:d}_cv_{cvfold}.mat'

    dir_pth = set_paths(representation_pth=representation_pth,exp_name=exp_name)

    CV = sio.loadmat(dir_pth['cvfolds']+cvfold_fname, squeeze_me=True)
    O = sio.loadmat(dir_pth['data']+'PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat', squeeze_me=True)

    #Fit GMMs on data that is reconstructed well
    RT = np.mean((CV['XrT'] - O['T_dat'])**2,axis=1)
    RE = np.nanmean((CV['XrE'] - np.concatenate([O['E_pc_scaled'],O['E_feature']],axis = 1))**2,axis=1)
    CTE= np.mean((CV['zT'] - CV['zE'])**2,axis=1)
    keep = np.logical_and(RT<np.percentile(RT,perc),RE<np.percentile(RE,perc))
    keep = np.logical_and(keep,CTE<np.percentile(CTE,perc))
    CV['train_ind'] = keep
    
    Z_train = np.concatenate([CV[fi][CV['train_ind'],:] for fi in fiton])
    Z_val   = np.concatenate([CV[fi][CV['val_ind'],:] for fi in fiton])
    
    #Initialize
    write_header=True
    bic_train = np.empty(n_components_range.size)
    bic_val = np.empty(n_components_range.size)

    for i,n_components in enumerate(n_components_range):
    
        #Declare GMM object
        gmm = mixture.GaussianMixture(n_components=n_components,
                                covariance_type='full',reg_covar=1e-04,
                                tol=tol,max_iter=max_iter,n_init=n_init,
                                init_params='kmeans',
                                random_state=None,
                                warm_start=False,
                                verbose=1)
        
        #Fit gmm + calculate bic
        gmm.fit(Z_train)
        bic_train[i] = gmm.bic(Z_train)
        bic_val[i] = gmm.bic(Z_val)

        #Write logs
        log_fname = fname+f'minn_{min_component:02d}_maxn_{max_component:02d}.csv'
        with open(dir_pth['result']+log_fname, "a") as logfile:
            writer = csv.writer(logfile, delimiter=',')
            if write_header:
                writer.writerow(['n_components','cv','bic_train','bic_val','converged'])
                write_header=False
            writer.writerow([n_components,cvfold,bic_train[i],bic_val[i],int(gmm.converged_)])

        #Save fitted gmm object
        fit_fname = fname+'_n_{:d}.pkl'.format(n_components)
        with open(dir_pth['result']+fit_fname, 'wb') as fid:
            pickle.dump(gmm, fid)
        print('Completed {:s} component model for cv {}'.format(str(n_components),cvfold))
        print(dir_pth['result']+fit_fname)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

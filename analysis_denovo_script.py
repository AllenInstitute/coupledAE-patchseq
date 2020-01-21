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

def set_paths(exp_name='denovo_clustering'):
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
parser.add_argument("--min_component",  default=10,            type=int,  help='min GMM components')
parser.add_argument("--max_component",  default=50,            type=int,  help='max GMM components')
parser.add_argument("--exp_name", default='denovo_clustering', type=str,  help='Result folder')

def main(min_component=10, max_component=50,
         exp_name='denovo_clustering'):
    
    #Representations
    alpha_T=1.0
    alpha_E=1.0
    alpha_M=1.0
    lambda_TE=0.0
    fiton='zT'

    #GMM parameters
    n_components_range = np.arange(min_component,max_component)
    n_cvfolds = 9
    n_init = 100
    max_iter = int(1e4)
    tol = 1e-6
    
    dir_pth = set_paths(exp_name=exp_name)
    
    #Load all CV sets 
    X_train = []
    XT = []
    XE = []
    for cvfold in range(9):
        cvfold_fname='v2_aT_'+str(alpha_T)+\
                    '_aE_'+str(alpha_E)+\
                    '_aM_'+str(alpha_M)+\
                    '_cs_'+str(lambda_TE)+\
                    '_ld_3_bs_200_se_500_ne_1500_cv_'+str(cvfold)+\
                    '_ri_0500_ft-summary'
        cvfold_fname=cvfold_fname.replace('.','-')+'.mat'
        CV = sio.loadmat(dir_pth['cvfolds']+cvfold_fname,squeeze_me=True)
        X_train.append(CV[fiton][CV['train_ind'],:])
        XT.append(CV['zT'])
        XE.append(CV['zE'])

    #Initialize
    write_header=True
    bic = np.empty((n_components_range.size,n_cvfolds))
    aic = np.empty((n_components_range.size,n_cvfolds))
    for i,n_components in enumerate(n_components_range):
        for cv in range(n_cvfolds):
            #Declare GMM object
            gmm = mixture.GaussianMixture(n_components=n_components,
                                    covariance_type='full',reg_covar=1e-04,
                                    tol=tol,max_iter=max_iter,n_init=n_init,
                                    init_params='kmeans',
                                    random_state=None,
                                    warm_start=False,
                                    verbose=0)

            #Fit gmm + calculate aic and bic
            gmm.fit(X_train[cv])
            bic[i,cv] = gmm.bic(X_train[cv])
            aic[i,cv] = gmm.aic(X_train[cv])

            #Write logs
            log_fname = 'logs_{:02d}_{:02d}.csv'.format(min_component,max_component)
            with open(dir_pth['result']+log_fname, "a") as logfile:
                writer = csv.writer(logfile, delimiter=',')
                if write_header:
                    writer.writerow(['n_components','cv','bic','aic','converged'])
                    write_header=False
                writer.writerow([n_components,cv,bic[i,cv],aic[i,cv],int(gmm.converged_)])

            #Save fitted gmm object
            fname = 'gmm_{:d}_comp_{:d}_cv.pkl'.format(n_components,cv)
            with open(dir_pth['result']+fname, 'wb') as fid:
                pickle.dump(gmm, fid)
        print('Completed {:s} component model'.format(str(n_components)))
    return 


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

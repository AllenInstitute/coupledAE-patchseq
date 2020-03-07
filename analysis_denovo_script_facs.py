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

def set_paths(exp_name='denovo_clustering_facs_imb'):
    """Set data paths
    """

    from pathlib import Path   
    dir_pth = {}
    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/'

    dir_pth['representation'] = base_path + 'dat/result/FACS_tests/'
    dir_pth['result'] = dir_pth['representation'] + exp_name + '/'

    Path(dir_pth['result']).mkdir(parents=True, exist_ok=True)
    return dir_pth

parser = argparse.ArgumentParser()
parser.add_argument("--min_component",  default=10,            type=int,  help='min GMM components')
parser.add_argument("--max_component",  default=50,            type=int,  help='max GMM components')
parser.add_argument("--max_samples",    default=100,            type=int,  help='max samples as listed on the summary file')
parser.add_argument("--exp_name", default='denovo_clustering_ldtest', type=str,  help='Result folder')

def main(min_component=10, max_component=50, max_samples=100,
         exp_name='denovo_clustering_ldtest'):
    
    #Representations
    fiton='zT'
    n_cvfolds = 1

    #GMM parameters
    n_components_range = np.arange(min_component,max_component)
    n_init = 100
    max_iter = int(1e4)
    tol = 1e-6
    
    dir_pth = set_paths(exp_name=exp_name)
    
    #Load all CV sets 
    X_train = []
    XT = []   

    #D = sio.loadmat(dir_pth['representation']+'apprx_balanced_2500_ld_3_bs_200_mins_0_maxs_{}_rs_0_se_500_ne_1500_ri_0100_ft-summary.mat'.format(max_samples),squeeze_me=True)
    D = sio.loadmat(dir_pth['representation']+'apprx_balanced_ldtest_ld_5_bs_200_mins_0_maxs_{}_rs_0_se_500_ne_1500_ri_0100_ft-summary.mat'.format(max_samples),squeeze_me=True)
    
    CV = D.copy()
    X_train.append(CV[fiton][CV['train_ind'],:])
    XT.append(CV['zT'])

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
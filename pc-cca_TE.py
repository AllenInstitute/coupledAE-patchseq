import argparse
import numpy as np
import scipy.io as sio
from ae_model_train_v3 import TE_get_splits
from refactor.utils.compute import CCA_extended 
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("--cvfold",            default=0,          type=int,     help="20 fold CV sets (range from 0 to 19)")
parser.add_argument("--Edat",              default='pcipfx',   type=str,     help="only implemented for `pcifpx`")
parser.add_argument("--pc_dim_T",          default=20,         type=int,     help="Number of pca dims for T data")
parser.add_argument("--pc_dim_E",          default=20,         type=int,     help="Number of pca dims for E data")
parser.add_argument("--cca_dim",           default=3,          type=int,     help="Number of cca (latent) dims")
parser.add_argument("--model_id",          default='PCCCA',    type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='TE_CS',    type=str,     help="Experiment set")


def set_paths(exp_name='TEMP'):
    from pathlib import Path   
    dir_pth = {}
    curr_path = str(Path().absolute())
    base_path = None
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'
    elif '/home/rohan' in curr_path:
        #base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
        base_path = '/home/rohan/Remote-AI/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'

    dir_pth['result'] =     base_path + 'dat/result/' + exp_name + '/'
    Path(dir_pth['result']).mkdir(parents=True, exist_ok=True) 
    return dir_pth


def mse(x, y): return np.mean((x-y)**2)

def main(cvfold=0, Edat='pcipfx', pc_dim_T=20, pc_dim_E=20, cca_dim=3, model_id='PCCCA', exp_name='TE_CS'):

    dir_pth = set_paths(exp_name=exp_name)

    fileid = (model_id + \
            f'_Edat_{Edat:s}' + \
            f'_pcT_{pc_dim_T:d}_pcE_{pc_dim_E:d}_cca_{cca_dim:d}' + \
            f'_cv_{cvfold:d}')

    if Edat == 'pcipfx': Edat = 'E_pcipxf'
    else: ValueError('Edat must be pcipfx')
    
    #Data definitions:
    D = sio.loadmat(dir_pth['data']+'PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat',squeeze_me=True)
    D['E_pcipxf'] = np.concatenate([D['E_pc_scaled'],D['E_feature']],axis = 1)
    D['E_pcipxf'][np.isnan(D['E_pcipxf'])]=0
    XT = D['T_dat']
    XE = D['E_pcipxf']

    train_ind,val_ind = TE_get_splits(matdict=D,cvfold=cvfold,n=20)

    #Reduce dims of T data
    pcaT = PCA(n_components=pc_dim_T)
    pcaT.fit_transform(XT[train_ind,:])
    XTpc = pcaT.transform(XT)

    #Reduce dims of E data
    pcaE = PCA(n_components=pc_dim_E)
    pcaE.fit_transform(XE[train_ind,:])
    XEpc = pcaE.transform(XE)

    #CCA on T and E data
    cca = CCA_extended(n_components=cca_dim, scale=True, max_iter=1e4, tol=1e-06, copy=True)
    cca.fit(XTpc[train_ind,:],XEpc[train_ind,:])
    zTcca,zEcca = cca.transform(XTpc,XEpc)
        
    #Within modality reconstruction
    XrTpc,XrEpc = cca.inverse_transform_xy(zTcca,zEcca)
    XrT = pcaT.inverse_transform(XrTpc)
    XrE = pcaE.inverse_transform(XrEpc)

    #Cross modality reconstruction
    XrTpc_from_zE, XrEpc_from_zT = cca.inverse_transform_xy(zEcca,zTcca)
    XrT_from_XE = pcaT.inverse_transform(XrTpc_from_zE)
    XrE_from_XT = pcaE.inverse_transform(XrEpc_from_zT)

    savemat = {'zT': zTcca,
               'zE': zEcca,
               'XrT': XrT,
               'XrE': XrE,
               'XrE_from_XT': XrE_from_XT,
               'XrT_from_XE': XrT_from_XE,
               'train_ind':train_ind,
               'val_ind':val_ind}

    print(f'Training '+\
          f'mseT {mse(XT[train_ind,:],XrT[train_ind,:]):0.3f} '+\
          f'mseE: {mse(XE[train_ind,:],XrE[train_ind,:]):0.3f} '+\
          f'mseTE: {mse(zTcca[train_ind,:],zEcca[train_ind,:]):0.3f}')

    print(f'Validation '+\
          f'mseT {mse(XT[val_ind,:],XrT[val_ind,:]):0.3f} '+\
          f'mseE: {mse(XE[val_ind,:],XrE[val_ind,:]):0.3f} '+\
          f'mseTE: {mse(zTcca[val_ind,:],zEcca[val_ind,:]):0.3f}')
    
    sio.savemat(dir_pth['result']+fileid+'.mat',savemat,do_compression=True)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

#Model updated for TF2.0

# 1. T and E data are matched 
# 2. Updated model can use cross modal reconstruction loss (set with flag)

# echo "python -m ae_model_train_v2 "\
#       "--batchsize 100 "\
#       "--cvfold 0 "\
#       "--Edat ipfx "\
#       "--alpha_T 1.0 "\
#       "--alpha_E 1.0 "\
#       "--lambda_TE 0.0"\
#       "--augment_decoders 0 "\
#       "--latent_dim 3 "\
#       "--n_epochs 10 "\
#       "--n_steps_per_epoch 500 "\
#       "--ckpt_save_freq 1000 "\
#       "--n_finetuning_steps 10 "\
#       "--run_iter 0 "\
#       "--model_id TEST "\
#       "--exp_name TEST"

import argparse
import csv
import os
import pdb

import numpy as np
import scipy.io as sio
import tensorflow as tf
from timebudget import timebudget

from ae_model_def import Model_TE_aug_decoders
from data_funcs import TE_get_splits_5, TE_get_splits_45

parser = argparse.ArgumentParser()

parser.add_argument("--batchsize",         default=200,                     type=int,     help="Batch size")
parser.add_argument("--cvfold",            default=0,                       type=int,     help="45 fold CV sets (range from 0 to 44)")
parser.add_argument("--Edat",              default='pcipfx',                type=str,     help="`pc` or `ipfx` or `pcifpx`")
parser.add_argument("--alpha_T",           default=1.0,                     type=float,   help="T Reconstruction loss weight")
parser.add_argument("--alpha_E",           default=1.0,                     type=float,   help="E Reconstruction loss weight")
parser.add_argument("--lambda_TE",         default=1.0,                     type=float,   help="Coupling loss weight")
parser.add_argument("--augment_decoders",  default=1,                       type=int,     help="0 or 1 - Train with cross modal reconstruction")
parser.add_argument("--latent_dim",        default=3,                       type=int,     help="Number of latent dims")

parser.add_argument("--n_epochs",          default=1500,                    type=int,     help="Number of epochs to train")
parser.add_argument("--n_steps_per_epoch", default=500,                     type=int,     help="Number of model updates per epoch")
parser.add_argument("--ckpt_save_freq",    default=100,                     type=int,     help="Frequency of checkpoint saves")
parser.add_argument("--n_finetuning_steps",default=100,                     type=int,     help="Number of fine tuning steps for E agent")

parser.add_argument("--run_iter",          default=0,                       type=int,     help="Run-specific id")
parser.add_argument("--model_id",          default='NM',                    type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='TE_NM',                 type=str,     help="Experiment set")

def set_paths(exp_name='Test'):
    """Set data and results path for network training. 

    Args:
        exp_name (str): Folder name to save results. Defaults to 'Test'.

    Returns:
        dir_pth (dict): dictionary with paths for logs, checkpoints, 
    """
    from pathlib import Path   

    dir_pth = {}
    current_path = Path().absolute()
    dir_pth['data'] = current_path / 'data/'
    dir_pth['result'] = current_path / 'results/' / exp_name
    dir_pth['checkpoint'] = dir_pth['result'] / 'checkpoints'
    dir_pth['logs'] = dir_pth['result'] / 'logs'
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True)
    Path(dir_pth['checkpoint']).mkdir(parents=True, exist_ok=True)
    return dir_pth

def main(batchsize=200, cvfold=0, Edat = 'pcifpx',
         alpha_T=1.0,alpha_E=1.0,lambda_TE=1.0,augment_decoders=True,
         latent_dim=3,n_epochs=1500, n_steps_per_epoch=500, ckpt_save_freq=100,
         n_finetuning_steps=100,
         run_iter=0, model_id='NM', exp_name='TE_NM'):
    
    dir_pth = set_paths(exp_name=exp_name)

    #Augmenting only hurts of networks are not coupled.
    if lambda_TE==0.0:
        augment_decoders=0

    fileid = model_id + \
        '_Edat_' + str(Edat) + \
        '_aT_' + str(alpha_T) + \
        '_aE_' + str(alpha_E) + \
        '_cs_' + str(lambda_TE) + \
        '_ad_' + str(augment_decoders) + \
        '_ld_' + str(latent_dim) + \
        '_bs_' + str(batchsize) + \
        '_se_' + str(n_steps_per_epoch) +\
        '_ne_' + str(n_epochs) + \
        '_cv_' + str(cvfold) + \
        '_ri_' + str(run_iter)
    fileid = fileid.replace('.', '-')

    #Convert to boolean
    augment_decoders=augment_decoders>0

    if Edat == 'pc':
        Edat = 'E_pc_scaled'
    elif Edat == 'ipfx':
        Edat = 'E_feature'
    elif Edat == 'pcipfx':
        Edat = 'E_pcipxf'
    else:
        raise ValueError('Edat must be spc or ipfx!')
 
    #Data operations and definitions:
    D = sio.loadmat(dir_pth['data']+'PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat',squeeze_me=True)
    D['E_pcipxf'] = np.concatenate([D['E_pc_scaled'],D['E_feature']],axis = 1)
    #train_ind,val_ind,test_ind = TE_get_splits_45(matdict=D,cvfold=cvfold)
    train_ind,val_ind,test_ind = TE_get_splits_5(matdict=D,cvfold=cvfold)

    Partitions = {'train_ind':train_ind,'val_ind':val_ind,'test_ind':test_ind}

    train_T_dat = D['T_dat'][train_ind,:]
    train_E_dat = D[Edat][train_ind,:]

    val_T_dat = D['T_dat'][val_ind,:]
    val_E_dat = D[Edat][val_ind,:]
    
    Edat_var = np.nanvar(D[Edat],axis=0)
    maxsteps = tf.constant(n_epochs*n_steps_per_epoch)
    batchsize = tf.constant(batchsize)
    
    #Model definition
    optimizer_main = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_generator = tf.data.Dataset.from_generator(Datagen,output_types=(tf.float32, tf.float32),
                                                     args=(maxsteps,batchsize,train_T_dat,train_E_dat))
    
    model_TE = Model_TE_aug_decoders(T_dim=train_T_dat.shape[1],
                        E_dim=train_E_dat.shape[1],
                        T_intermediate_dim=50,
                        E_intermediate_dim=40,
                        T_dropout=0.5,
                        E_gauss_noise_wt=Edat_var,
                        E_gnoise_sd=0.05,
                        E_dropout=0.1,
                        alpha_T = alpha_T,
                        alpha_E = alpha_E,
                        lambda_TE = lambda_TE,
                        latent_dim=latent_dim,
                        name='TE')

    #Model training functions 
    @tf.function
    def train_fn(model, optimizer, XT, XE, train_T=False, train_E=False, augment_decoders=True, subnetwork='all'):
        """Enclose this with tf.function to create a fast training step. Function can be used for inference as well. 
        
        Args:
            XT: T data for training or validation
            XE: E data for training or validation
            train_T: {bool} -- Switch augmentation for T data on or off
            train_E {bool} -- Switch augmentation for E data on or off
            subnetwork {str} -- 'all' or 'E'. 'all' trains the full network, 'E' trains only the E arm.
        """
        
        with tf.GradientTape() as tape:
            zT, zE, XrT, XrE = model((XT, XE), 
                                    train_T=train_T, 
                                    train_E=train_E,
                                    augment_decoders=augment_decoders)
            
            #Apply updates to specified subnetworks:
            if subnetwork is 'all':
                trainable_weights = [weight for weight in model.trainable_weights]
            elif subnetwork is 'E':
                trainable_weights = [weight for weight in model.trainable_weights if '_E' in weight.name]
            
            loss = sum(model.losses)

        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        return zT, zE, XrT, XrE

    #Model logging functions

    def report_losses(model, epoch, datatype='train', verbose=False):
        mse_loss_T = model.mse_loss_T.numpy()
        mse_loss_E = model.mse_loss_E.numpy()
        mse_loss_TE = model.mse_loss_TE.numpy()
        if verbose:
            print('Epoch:{:5d}, '
                    'mse_T: {:0.3f}, '
                    'mse_E: {:0.3f}, '
                    'mse_TE: {:0.3f}'.format(epoch,
                                            mse_loss_T,
                                            mse_loss_E,
                                            mse_loss_TE))

        log_name = [datatype+i for i in ['epoch','mse_T', 'mse_E', 'mse_TE']]
        log_values = [epoch, mse_loss_T, mse_loss_E, mse_loss_TE]
        return log_name, log_values
    
    def save_results(this_model,Data,fname,Inds=Partitions,Edat=Edat):
        all_T_dat = tf.constant(Data['T_dat'])
        all_E_dat = tf.constant(Data[Edat])
        zT, zE, XrT, XrE = this_model((all_T_dat, all_E_dat), training=False)
        XrE_from_XT = this_model.decoder_E(zT, training=False)
        XrT_from_XE = this_model.decoder_T(zE, training=False)

        savemat = {'zT': zT.numpy(),
                'zE': zE.numpy(),
                'XrE': XrE.numpy(),
                'XrE_from_XT': XrE_from_XT.numpy(),
                'XrT': XrT.numpy(),
                'XrT_from_XE': XrT_from_XE.numpy()}
        
        savemat.update(Inds)
        sio.savemat(fname, savemat, do_compression=True)
        return

    #Main training loop ----------------------------------------------------------------------
    epoch=0
    for step, (XT,XE) in enumerate(train_generator): 
        train_fn(model=model_TE, optimizer=optimizer_main, XT=XT, XE=XE, 
                 train_T=True,train_E=True,augment_decoders=augment_decoders,subnetwork='all')
        
        if (step+1) % n_steps_per_epoch == 0:
            #Update epoch count
            epoch = epoch+1

            #Collect training metrics
            model_TE((train_T_dat, train_E_dat), train_T=False, train_E=False)
            train_log_name, train_log_values = report_losses(model=model_TE ,epoch=epoch, datatype='train_', verbose=True)
            
            #Collect validation metrics
            model_TE((val_T_dat, val_E_dat), train_T=False, train_E=False)
            val_log_name, val_log_values = report_losses(model=model_TE, epoch=epoch, datatype='val_', verbose=True)
            
            with open(dir_pth['logs']+fileid+'.csv', "a") as logfile:
                writer = csv.writer(logfile, delimiter=',')
                #Write headers to the log file
                if epoch == 1:
                    writer.writerow(train_log_name+val_log_name)
                writer.writerow(train_log_values+val_log_values)

            if epoch % ckpt_save_freq == 0:
                #Save model weights
                model_TE.save_weights(dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-weights.h5')
                #Save reconstructions and results for the full dataset:
                save_results(this_model=model_TE,Data=D,fname=dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-summary.mat')
            
    #Save model weights on exit
    model_TE.save_weights(dir_pth['result']+fileid+'-weights.h5')
    
    #Save reconstructions and results for the full dataset:
    save_results(this_model=model_TE,Data=D,fname=dir_pth['result']+fileid+'-summary.mat')

    print('\n\n--- fine tuning loop begins ---')
    #Fine tuning loop ----------------------------------------------------------------------
    #Each batch is now the whole training set
    for epoch in range(n_finetuning_steps):
        train_fn(model=model_TE, optimizer=optimizer_main, XT=train_T_dat, XE=train_E_dat, 
                 train_T=True,train_E=True,augment_decoders=augment_decoders,subnetwork='all')
        
        #Collect training metrics
        model_TE((train_T_dat, train_E_dat), train_T=False, train_E=False)
        train_log_name, train_log_values = report_losses(model=model_TE ,epoch=epoch, datatype='train_', verbose=True)
        
        #Collect validation metrics
        model_TE((val_T_dat, val_E_dat), train_T=False, train_E=False)
        val_log_name, val_log_values = report_losses(model=model_TE, epoch=epoch, datatype='val_', verbose=True)

        with open(dir_pth['logs']+fileid+'_'+str(n_finetuning_steps)+'_ft.csv', "a") as logfile:
            writer = csv.writer(logfile, delimiter=',')
            #Write headers to the log file
            if epoch == 0:
                writer.writerow(train_log_name+val_log_name)
            writer.writerow(train_log_values+val_log_values)
    
    #Save model weights on exit
    model_TE.save_weights(dir_pth['result']+fileid+'_'+str(n_finetuning_steps)+'_ft-weights.h5')

    #Save reconstructions and results for the full dataset:
    save_results(this_model=model_TE,Data=D,fname=dir_pth['result']+fileid+'_'+str(n_finetuning_steps)+'_ft-summary.mat')    
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

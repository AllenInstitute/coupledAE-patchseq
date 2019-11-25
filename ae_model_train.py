#Model updated for TF2.0
#python -m ae_model_train --batchsize 100 --cvfold 0 --alpha_T 1.0 --alpha_E 1.0 --alpha_M 1.0 --lambda_TE 0.0 --latent_dim 3 --n_epochs 2000 --n_steps_per_epoch 500 --ckpt_save_freq 100 --run_iter 0 --model_id 'v1' --exp_name 'TE_Patchseq_Bioarxiv'
import argparse
import os
import pdb
import re
import socket
import sys
import timeit

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers
from data_funcs import TEM_get_splits
from ae_model_def import Model_TE
import csv
from timebudget import timebudget

parser = argparse.ArgumentParser()

parser.add_argument("--batchsize",         default=200,                     type=int,     help="Batch size")
parser.add_argument("--cvfold",            default=0,                       type=int,     help="9 fold CV sets (range from 0 to 8)")

parser.add_argument("--alpha_T",           default=1.0,                     type=float,   help="T Reconstruction loss weight")
parser.add_argument("--alpha_E",           default=0.1,                     type=float,   help="E Reconstruction loss weight")
parser.add_argument("--alpha_M",           default=1.0,                     type=float,   help="M Reconstruction loss weight")
parser.add_argument("--lambda_TE",         default=1.0,                     type=float,   help="Coupling loss weight")

parser.add_argument("--latent_dim",        default=3,                       type=int,     help="Number of latent dims")

parser.add_argument("--n_epochs",          default=1500,                    type=int,     help="Number of epochs to train")
parser.add_argument("--n_steps_per_epoch", default=500,                     type=int,     help="Number of model updates per epoch")
parser.add_argument("--ckpt_save_freq",    default=100,                     type=int,     help="Frequency of checkpoint saves")

parser.add_argument("--run_iter",          default=0,                       type=int,     help="Run-specific id")
parser.add_argument("--model_id",          default='v1',                    type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='TE_Patchseq_Bioarxiv',  type=str,     help="Experiment set")

def set_paths(exp_name='TEMP'):
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

    dir_pth['result'] =     base_path + 'dat/result/' + exp_name + '/'
    dir_pth['checkpoint'] = dir_pth['result'] + 'checkpoints/'
    dir_pth['logs'] =       dir_pth['result'] + 'logs/'

    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True) 
    Path(dir_pth['checkpoint']).mkdir(parents=True, exist_ok=True) 
    return dir_pth


class Datagen():
    """Iterator class to sample the dataset. Tensors T_dat and E_dat are provided at runtime.
    """
    def __init__(self, maxsteps, batchsize, T_dat, E_dat):
        self.T_dat = T_dat
        self.E_dat = E_dat
        self.batchsize = batchsize
        self.maxsteps = maxsteps
        self.n_samples = self.T_dat.shape[0]
        self.count = 0
        print('Initializing generator...')
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.maxsteps:
            self.count = self.count+1
            ind = np.random.randint(0, self.n_samples, self.batchsize)
            return (self.T_dat[ind, :], self.E_dat[ind, :])
        else:
            raise StopIteration

def main(batchsize=200, cvfold=0,
         alpha_T=1.0,alpha_E=1.0,alpha_M=1.0,lambda_TE=0.0,
         latent_dim=3,n_epochs=1500, n_steps_per_epoch=500, ckpt_save_freq=100,
         run_iter=0, model_id='v1', exp_name='TE_Patchseq_Bioarxiv'):
    
    dir_pth = set_paths(exp_name=exp_name)
    fileid = model_id + \
        '_aT_' + str(alpha_T) + \
        '_aE_' + str(alpha_E) + \
        '_aM_' + str(alpha_M) + \
        '_cs_' + str(lambda_TE) + \
        '_ld_' + str(latent_dim) + \
        '_bs_' + str(batchsize) + \
        '_se_' + str(n_steps_per_epoch) +\
        '_ne_' + str(n_epochs) + \
        '_cv_' + str(cvfold) + \
        '_ri_' + str(run_iter)
    fileid = fileid.replace('.', '-')

    #Load data:
    D = sio.loadmat(dir_pth['data']+'PS_v4_beta_0-4_matched_well-sampled.mat',squeeze_me=True)
    cvset,testset = TEM_get_splits(D)

    train_ind = cvset[cvfold]['train']
    train_T_dat = tf.constant(D['T_dat'][train_ind,:])
    train_E_dat = D['E_dat'][train_ind,:]
    train_M_dat = D['M_dat'][train_ind]
    train_E_dat = tf.constant(np.concatenate([train_E_dat,train_M_dat.reshape(train_M_dat.size,1)],axis=1))

    val_ind = cvset[cvfold]['val']
    val_T_dat = D['T_dat'][val_ind,:]
    val_E_dat = D['E_dat'][val_ind,:]
    val_M_dat = D['M_dat'][val_ind]
    val_E_dat = tf.constant(np.concatenate([val_E_dat,val_M_dat.reshape(val_M_dat.size,1)],axis=1))
    Xval = (tf.constant(val_T_dat),tf.constant(val_E_dat))

    maxsteps = tf.constant(n_epochs*n_steps_per_epoch)
    batchsize = tf.constant(batchsize)
    alpha_T   = tf.constant(alpha_T,dtype=tf.float32)
    alpha_E   = tf.constant(alpha_E,dtype=tf.float32)
    alpha_M   = tf.constant(alpha_M,dtype=tf.float32)
    lambda_TE = tf.constant(lambda_TE,dtype=tf.float32)

    def min_var_loss(zi, zj, Wij=None):
        """SVD is calculated over entire batch. MSE is calculated over only paired entries within batch
        """
        batch_size = tf.shape(zi)[0]
        if Wij is None:
            Wij_ = tf.ones([batch_size, ])
        else:
            Wij_ = tf.reshape(Wij, [batch_size, ])

        zi_paired = tf.boolean_mask(zi, tf.math.greater(Wij_, 1e-2))
        zj_paired = tf.boolean_mask(zj, tf.math.greater(Wij_, 1e-2))
        Wij_paired = tf.boolean_mask(Wij_, tf.math.greater(Wij_, 1e-2))

        vars_j_ = tf.square(tf.linalg.svd(zj - tf.reduce_mean(zj, axis=0), compute_uv=False))/tf.cast(batch_size - 1, tf.float32)
        vars_j  = tf.where(tf.math.is_nan(vars_j_), tf.zeros_like(vars_j_) + tf.cast(1e-2,dtype=tf.float32), vars_j_)
        weighted_distance = tf.multiply(tf.sqrt(tf.reduce_sum(tf.math.squared_difference(zi_paired, zj_paired),axis=1)),Wij_paired)
        loss_ij    = tf.reduce_mean(weighted_distance,axis=None)/tf.maximum(tf.reduce_min(vars_j, axis=None),tf.cast(1e-2,dtype=tf.float32))
        return loss_ij

    def report_losses(XT, XE, zT, zE, XrT, XrE, datatype='train', verbose=False):
        mse_loss_T = tf.reduce_mean(tf.math.squared_difference(XT, XrT))
        mse_loss_E = tf.reduce_mean(tf.math.squared_difference(XE, XrE))
        mse_loss_M = tf.reduce_mean(tf.math.squared_difference(XE[:, -1], XrE[:, -1]))
        mse_loss_TE = tf.reduce_mean(tf.math.squared_difference(zT, zE))

        if verbose:
            print('Epoch:{:5d}, '
                  'mse_T: {:0.3f}, '
                  'mse_E: {:0.3f}, '
                  'mse_M: {:0.3f}, '
                  'mse_TE: {:0.3f}'.format(epoch,
                                           mse_loss_T.numpy(),
                                           mse_loss_E.numpy(),
                                           mse_loss_M.numpy(),
                                           mse_loss_TE.numpy()))

        log_name = [datatype+i for i in ['epoch','mse_T', 'mse_E', 'mse_M', 'mse_TE']]
        log_values = [epoch, mse_loss_T.numpy(), mse_loss_E.numpy(),
                      mse_loss_M.numpy(), mse_loss_TE.numpy()]
        return log_name, log_values

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_generator = tf.data.Dataset.from_generator(Datagen,output_types=(tf.float32, tf.float32),
                                                     args=(maxsteps,batchsize,train_T_dat,train_E_dat))

    model_TE = Model_TE(T_output_dim=train_T_dat.shape[1],
                        E_output_dim=train_E_dat.shape[1],
                        T_intermediate_dim=50,
                        E_intermediate_dim=40,
                        T_dropout=0.5,
                        E_gnoise_sd=0.05,
                        E_dropout=0.1,
                        latent_dim=latent_dim,
                        name='TE')

    epoch=0
    for step, x_batch in enumerate(train_generator):
        with tf.GradientTape() as tape:
            XT = x_batch[0]
            XE = x_batch[1]
            zT, zE, XrT, XrE = model_TE((XT, XE), training=True)
            mse_loss_T = tf.reduce_mean(tf.math.squared_difference(XT, XrT))
            mse_loss_E = tf.reduce_mean(tf.math.squared_difference(XE[:, :-1], XrE[:, :-1]))
            mse_loss_M = tf.reduce_mean(tf.math.squared_difference(XE[:, -1], XrE[:, -1]))
            cpl_loss_TE = min_var_loss(zT, zE)
            loss = alpha_T*mse_loss_T + \
                alpha_E*mse_loss_E + \
                alpha_M*mse_loss_M + \
                lambda_TE*cpl_loss_TE

        grads = tape.gradient(loss, model_TE.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_TE.trainable_weights))

        if (step+1) % n_steps_per_epoch == 0:
            #Update epoch count
            epoch = epoch+1

            #Report training metrics
            zT, zE, XrT, XrE = model_TE((train_T_dat, train_E_dat), training=False)
            train_log_name, train_log_values = report_losses(train_T_dat, train_E_dat, zT, zE, XrT, XrE, datatype='train_', verbose='True')

            #Report validation metrics
            zT, zE, XrT, XrE = model_TE((val_T_dat, val_E_dat), training=False)
            val_log_name, val_log_values = report_losses(val_T_dat, val_E_dat, zT, zE, XrT, XrE, datatype='val_', verbose='True')

            with open(dir_pth['logs']+fileid+'.csv', "a") as logfile:
                writer = csv.writer(logfile, delimiter=',')
                if epoch == 1:
                    writer.writerow(train_log_name+val_log_name)
                writer.writerow(train_log_values+val_log_values)

            if epoch % ckpt_save_freq == 0:
                #Save model weights
                model_TE.save_weights(dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-weights.h5')

                #Save reconstructions and results for the full dataset:
                all_T_dat = tf.constant(D['T_dat'])
                all_E_dat = D['E_dat']
                all_M_dat = D['M_dat']
                all_E_dat = tf.constant(np.concatenate([all_E_dat, all_M_dat.reshape(all_M_dat.size, 1)], axis=1))
                zT, zE, XrT, XrE = model_TE((all_T_dat, all_E_dat), training=False)
                XrE_from_XT = model_TE.decoder_E(zT, training=False)
                XrT_from_XE = model_TE.decoder_T(zE, training=False)

                savemat = {'zT': zT.numpy(),
                        'zE': zE.numpy(),
                        'XE': XE.numpy(),
                        'XrE': XrE.numpy(),
                        'XrE_from_XT': XrE_from_XT.numpy(),
                        'XT': XT.numpy(),
                        'XrT': XrT.numpy(),
                        'XrT_from_XE': XrT_from_XE.numpy(),
                        'train_ind': train_ind,
                        'val_ind': val_ind,
                        'test_ind': testset,
                        'cvset': cvset}

                sio.savemat(dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-summary.mat', savemat, do_compression=True)


    #Save model weights on exit
    model_TE.save_weights(dir_pth['result']+fileid+'-weights.h5')

    #Save reconstructions and results for the full dataset:
    all_T_dat = tf.constant(D['T_dat'])
    all_E_dat = D['E_dat']
    all_M_dat = D['M_dat']
    all_E_dat = tf.constant(np.concatenate([all_E_dat, all_M_dat.reshape(all_M_dat.size, 1)], axis=1))
    zT, zE, XrT, XrE = model_TE((all_T_dat, all_E_dat), training=False)
    XrE_from_XT = model_TE.decoder_E(zT, training=False)
    XrT_from_XE = model_TE.decoder_T(zE, training=False)

    savemat = {'zT': zT.numpy(),
               'zE': zE.numpy(),
               'XE': XE.numpy(),
               'XrE': XrE.numpy(),
               'XrE_from_XT': XrE_from_XT.numpy(),
               'XT': XT.numpy(),
               'XrT': XrT.numpy(),
               'XrT_from_XE': XrT_from_XE.numpy(),
               'train_ind': train_ind,
               'val_ind': val_ind,
               'test_ind': testset,
               'cvset': cvset}

    sio.savemat(dir_pth['result']+fileid+'-summary.mat', savemat, do_compression=True)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

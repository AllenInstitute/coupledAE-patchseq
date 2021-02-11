#Model updated for TF2.0

import argparse
import os
import pdb
import re
import socket
import sys
import timeit
import time


import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers
from data_funcs import labelwise_samples
from ae_model_def import Model_T
import csv
from timebudget import timebudget

parser = argparse.ArgumentParser()

parser.add_argument("--batchsize",         default=200,                     type=int,     help="Batch size")
parser.add_argument("--min_samples",       default=0,                       type=int,     help="Types with less than this are ignored")
parser.add_argument("--max_samples",       default=100,                     type=int,     help="Number of samples per type")
parser.add_argument("--data_sampler_seed", default=0,                       type=int,     help="Seed for randomly sampling dataset")

parser.add_argument("--latent_dim",        default=3,                       type=int,     help="Number of latent dims")

parser.add_argument("--n_epochs",          default=1500,                    type=int,     help="Number of epochs to train")
parser.add_argument("--n_steps_per_epoch", default=500,                     type=int,     help="Number of model updates per epoch")
parser.add_argument("--ckpt_save_freq",    default=100,                     type=int,     help="Frequency of checkpoint saves")
parser.add_argument("--n_finetuning_steps",default=100,                     type=int,     help="Number of fine tuning steps for E agent")

parser.add_argument("--run_iter",          default=0,                       type=int,     help="Run-specific id")
parser.add_argument("--model_id",          default='eq_samples',            type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='FACS_tests',            type=str,     help="Experiment set")

def set_paths(exp_name='TEMP'):
    from pathlib import Path   
    dir_pth = {}
    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
        dir_pth['data'] = base_path + 'dat/raw/'

    dir_pth['result'] =     base_path + 'dat/result/' + exp_name + '/'
    dir_pth['checkpoint'] = dir_pth['result'] + 'checkpoints/'
    dir_pth['logs'] =       dir_pth['result'] + 'logs/'

    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True) 
    Path(dir_pth['checkpoint']).mkdir(parents=True, exist_ok=True) 
    return dir_pth


class Datagen():
    """Iterator class to sample the dataset. Tensors T_dat and E_dat are provided at runtime.
    """
    def __init__(self, maxsteps, batchsize, T_dat):
        self.T_dat = T_dat
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
            return (self.T_dat[ind, :])
        else:
            raise StopIteration

def main(batchsize=200, min_samples=0, max_samples=100, data_sampler_seed=0,
         latent_dim=3,n_epochs=1500, n_steps_per_epoch=500, ckpt_save_freq=100,
         n_finetuning_steps=100,
         run_iter=0, model_id='eq_samples', exp_name='FACS_tests'):
    
    dir_pth = set_paths(exp_name=exp_name)
    fileid = model_id + \
        '_ld_' + str(latent_dim) + \
        '_bs_' + str(batchsize) + \
        '_mins_' + str(min_samples) + \
        '_maxs_' + str(max_samples) + \
        '_rs_' + str(data_sampler_seed) + \
        '_se_' + str(n_steps_per_epoch) +\
        '_ne_' + str(n_epochs) + \
        '_ri_' + str(run_iter)
    fileid = fileid.replace('.', '-')

    #Load data:
    D = sio.loadmat(dir_pth['data']+'Mouse-V1-ALM-20180520_GABA_patchseq_v1.mat',squeeze_me=True)    
    train_ind = labelwise_samples(labels=D['cluster'].copy(),
                                  min_samples=min_samples,
                                  max_samples=max_samples,
                                  random_seed=data_sampler_seed)

    train_T_dat = tf.constant(D['log1p'][train_ind,:])
    val_ind = train_ind
    val_T_dat = train_T_dat
    
    maxsteps = tf.constant(n_epochs*n_steps_per_epoch)
    batchsize = tf.constant(batchsize)
    
    def report_losses(XT, XrT, epoch, datatype='train', verbose=False):
        mse_loss_T = tf.reduce_mean(tf.math.squared_difference(XT, XrT))
        if verbose: print('Epoch:{:5d}, mse_T: {:0.3f}, '.format(epoch,mse_loss_T.numpy()))
        log_name = [datatype+i for i in ['epoch','mse_T']]
        log_values = [epoch, mse_loss_T.numpy()]
        return log_name, log_values

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_generator = tf.data.Dataset.from_generator(Datagen,output_types=(tf.float32),
                                                     args=(maxsteps,batchsize,train_T_dat))

    model_T = Model_T(T_output_dim=train_T_dat.shape[1],
                      T_intermediate_dim=50,
                      T_dropout=0.5,
                      latent_dim=latent_dim,
                      name='T')

    @tf.function
    def train_fn(XT, train_T=False, subnetwork='all'):
        """Enclose this with tf.function to create a fast training step. Function can be used for inference as well. 
        Arguments:
            XT: T data for training or validation
            train_T: {bool} -- Switch augmentation for T data on or off
            subnetwork {str} -- 'all' or None. Use None for eval mode.
        """
        with tf.GradientTape() as tape:
            zT, XrT = model_T(XT, train_T=train_T)
            
            #Find the weights to update
            mse_loss_T = tf.reduce_mean(tf.math.squared_difference(XT, XrT))
            loss = mse_loss_T

            #Apply updates to specified subnetworks:
            if subnetwork is 'all':
                trainable_weights = [weight for weight in model_T.trainable_weights]
                grads = tape.gradient(loss, trainable_weights)
                optimizer.apply_gradients(zip(grads, trainable_weights))
                
        return zT, XrT

    def save_results(this_model, Data, fname):
        all_T_dat = tf.constant(Data['log1p'])
        zT, XrT = this_model(all_T_dat, training=False)

        savemat = {'zT': zT.numpy(),
                   'XT': XT.numpy(),
                   'XrT': XrT.numpy(),
                   'train_ind': train_ind,
                   'val_ind': val_ind}

        sio.savemat(fname, savemat, do_compression=True)
        return

    #Main training loop ----------------------------------------------------------------------
    epoch=0
    for step, XT in enumerate(train_generator): 
        zT, XrT = train_fn(XT=XT, train_T=True)
        
        if (step+1) % n_steps_per_epoch == 0:
            #Update epoch count
            epoch = epoch+1

            #Collect training metrics
            zT, XrT = train_fn(XT=train_T_dat, train_T=False, subnetwork=None)
            train_log_name, train_log_values = report_losses(train_T_dat, XrT, epoch, datatype='train_', verbose=True)
            
            #Collect validation metrics
            zT, XrT = train_fn(XT=val_T_dat, train_T=False, subnetwork=None)
            val_log_name, val_log_values = report_losses(val_T_dat, XrT, epoch, datatype='val_', verbose=True)

            with open(dir_pth['logs']+fileid+'.csv', "a") as logfile:
                writer = csv.writer(logfile, delimiter=',')
                #Write headers to the log file
                if epoch == 1:
                    writer.writerow(train_log_name+val_log_name)
                writer.writerow(train_log_values+val_log_values)

            if epoch % ckpt_save_freq == 0:
                #Save model weights
                model_T.save_weights(dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-weights.h5')
                #Save reconstructions and results for the full dataset:
                save_results(this_model=model_T,Data=D,fname=dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-summary.mat')
            
    #Save model weights on exit
    model_T.save_weights(dir_pth['result']+fileid+'-weights.h5')
    #Save reconstructions and results for the full dataset:
    save_results(this_model=model_T,Data=D,fname=dir_pth['result']+fileid+'-summary.mat')

    print('\n\n starting fine tuning loop')
    #Fine tuning loop ----------------------------------------------------------------------
    #Each batch is now the whole dataset
    
    for epoch in range(n_finetuning_steps):
        #Switch of T augmentation, and update only E arm:
        zT, XrT = train_fn(XT=train_T_dat, train_T=True,subnetwork='all')
        
        #Collect training metrics
        zT, XrT = train_fn(XT=train_T_dat, train_T=False,subnetwork=None)
        train_log_name, train_log_values = report_losses(train_T_dat,XrT, epoch, datatype='train_', verbose=True)
        
        #Collect validation metrics
        zT, XrT = train_fn(XT=val_T_dat, train_T=False, subnetwork=None)
        val_log_name, val_log_values = report_losses(val_T_dat, XrT, epoch, datatype='val_', verbose=True)

        with open(dir_pth['logs']+fileid+str(n_finetuning_steps)+'_ft.csv', "a") as logfile:
            writer = csv.writer(logfile, delimiter=',')
            #Write headers to the log file
            if epoch == 0:
                writer.writerow(train_log_name+val_log_name)
            writer.writerow(train_log_values+val_log_values)
    
    #Save model weights on exit
    model_T.save_weights(dir_pth['result']+fileid+str(n_finetuning_steps)+'_ft-weights.h5')
    #Save reconstructions and results for the full dataset:
    save_results(this_model=model_T,Data=D,fname=dir_pth['result']+fileid+str(n_finetuning_steps)+'_ft-summary.mat')    
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

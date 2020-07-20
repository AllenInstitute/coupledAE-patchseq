import tensorflow as tf
import numpy as np
import scipy.io as sio


class Datagen():
    """Iterator class to sample the dataset. Tensors T_dat and E_dat are provided at runtime.

    Args:
        maxsteps: length of generator
        batchsize: samples per batch
        T_dat: transcriptomic data matrix
        E_dat: electrophysiological data matrix
    """
    def __init__(self, maxsteps, batchsize, T_dat, E_dat):
        self.T_dat = T_dat
        self.E_dat = E_dat
        self.batchsize = batchsize
        self.maxsteps = maxsteps
        self.n_samples = self.T_dat.shape[0]
        self.count = 0
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.maxsteps:
            self.count = self.count+1
            ind = np.random.randint(0, self.n_samples, self.batchsize)
            return (tf.constant(self.T_dat[ind, :],dtype=tf.float32), 
                    tf.constant(self.E_dat[ind, :],dtype=tf.float32))
        else:
            raise StopIteration



D = sio.loadmat(dir_pth['data']+'PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat',squeeze_me=True)
D['E_pcipxf'] = np.concatenate([D['E_pc_scaled'],D['E_feature']],axis = 1)
#train_ind,val_ind,test_ind = TE_get_splits_45(matdict=D,cvfold=cvfold)
train_ind,val_ind,test_ind = TE_get_splits_5(matdict=D,cvfold=cvfold)
Partitions = {'train_ind':train_ind,'val_ind':val_ind,'test_ind':test_ind}
import argparse
import csv
import numpy as np
import scipy.io as sio
import tensorflow as tf
from ae_model_def_v3 import Model_TE_aug_decoders


parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",         default=200,                     type=int,     help="Batch size")
parser.add_argument("--cvfold",            default=0,                       type=int,     help="20 fold CV sets (range from 0 to 19)")
parser.add_argument("--Edat",              default='pcipfx',                type=str,     help="`pc` or `ipfx` or `pcifpx`")
parser.add_argument("--alpha_T",           default=1.0,                     type=float,   help="T Reconstruction loss weight")
parser.add_argument("--alpha_E",           default=1.0,                     type=float,   help="E Reconstruction loss weight")
parser.add_argument("--lambda_TE",         default=1.0,                     type=float,   help="Coupling loss weight")
parser.add_argument("--augment_decoders",  default=1,                       type=int,     help="0 or 1 : Train with cross modal reconstruction")
parser.add_argument("--latent_dim",        default=3,                       type=int,     help="Number of latent dims")
parser.add_argument("--n_epochs",          default=1500,                    type=int,     help="Number of epochs to train")
parser.add_argument("--n_steps_per_epoch", default=500,                     type=int,     help="Number of model updates per epoch")
parser.add_argument("--n_finetuning_steps",default=100,                     type=int,     help="Number of fine tuning steps with whole dataset")
parser.add_argument("--run_iter",          default=0,                       type=int,     help="Run-specific id")
parser.add_argument("--model_id",          default='CS',                    type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='TE_CS',                 type=str,     help="Experiment set")


def set_paths(exp_name='TEMP'):
    from pathlib import Path   
    dir_pth = {}
    curr_path = str(Path().absolute())
    base_path = None
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
    dir_pth['logs'] =       dir_pth['result'] + 'logs/'
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True) 
    return dir_pth


def TE_get_splits(matdict,cvfold,n=20):
    from sklearn.model_selection import StratifiedKFold
    """Obtain n=20 stratified splits of the data, as per the cluster labels. 
    Returns:
        cvset -- cvset id
    Arguments:
        matdict -- dataset dictionary
    """

    skf = StratifiedKFold(n_splits=n, random_state=0, shuffle=True)
    ind_dict = [{'train':train_ind, 'val':val_ind} for train_ind, val_ind in skf.split(X=np.zeros(shape=matdict['cluster'].shape), y=matdict['cluster'])]
    train_ind = ind_dict[cvfold]['train']
    val_ind   = ind_dict[cvfold]['val']
    return train_ind, val_ind


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


def main(batchsize=200, cvfold=0, Edat = 'pcifpx',
         alpha_T=1.0,alpha_E=1.0,lambda_TE=1.0,augment_decoders=True,
         latent_dim=3,n_epochs=1500, n_steps_per_epoch=500,
         n_finetuning_steps=100,
         run_iter=0, model_id='NM', exp_name='TE_NM'):
    
    dir_pth = set_paths(exp_name=exp_name)

    #Augmenting only makes sense when lambda_TE>0
    if lambda_TE==0.0:
        augment_decoders=0

    fileid = (model_id + \
            f'_Edat_{Edat:s}' + \
            f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_cs_{str(lambda_TE)}_ad_{str(augment_decoders)}' + \
            f'_ld_{latent_dim:d}_bs_{batchsize:d}_se_{n_steps_per_epoch:d}_ne_{n_epochs:d}' + \
            f'_cv_{cvfold:d}_ri_{run_iter:d}').replace('.', '-')

    #Convert int to boolean
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
    train_ind,val_ind = TE_get_splits(matdict=D,cvfold=cvfold,n=20)
    Partitions = {'train_ind':train_ind,'val_ind':val_ind}

    train_T_dat = D['T_dat'][train_ind,:]
    train_E_dat = D[Edat][train_ind,:]

    val_T_dat = D['T_dat'][val_ind,:]
    val_E_dat = D[Edat][val_ind,:]
    
    Edat_var = np.nanvar(D[Edat],axis=0)
    maxsteps = n_epochs*n_steps_per_epoch

    best_loss = 1e10
    
    #Model definition
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_generator = tf.data.Dataset.from_generator(Datagen,
                                                     output_types=(tf.float32, tf.float32),
                                                     args=(tf.constant(maxsteps), tf.constant(batchsize), train_T_dat, train_E_dat))
    
    model_TE = Model_TE_aug_decoders(T_dim=train_T_dat.shape[1],
                                     E_dim=train_E_dat.shape[1],
                                     T_intermediate_dim=50,
                                     E_intermediate_dim=40,
                                     T_dropout=0.5,
                                     E_gauss_noise_wt=Edat_var,
                                     E_gnoise_sd=0.05,
                                     E_dropout=0.1,
                                     alpha_T=alpha_T,
                                     alpha_E=alpha_E,
                                     lambda_TE=lambda_TE,
                                     latent_dim=latent_dim,
                                     train_E=True,
                                     train_T=True,
                                     augment_decoders=augment_decoders,
                                     name='TE')

    #Model training functions 
    @tf.function
    def train_fn(model, optimizer, XT, XE):
        """Enclose this with tf.function to create a fast training step. Function can be used for inference as well. 
        Arguments:
            XT: T data for training or validation
            XE: E data for training or validation
        """
        model.train_T=True
        model.train_E=True
        with tf.GradientTape() as tape:
            zT, zE, XrT, XrE = model((XT, XE))
            trainable_weights = [weight for weight in model.trainable_weights]
            loss = sum(model.losses)

        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        model.train_T=False
        model.train_E=False
        return zT, zE, XrT, XrE

    #Logging function
    def report_losses(model, epoch, datatype='train', verbose=False):
        mse_loss_T  = model.mse_loss_T.numpy()
        mse_loss_E  = model.mse_loss_E.numpy()
        mse_loss_TE = model.mse_loss_TE.numpy()
        if verbose:
            print(f'Epoch:{epoch:5d}, mse_T: {mse_loss_T:0.3f}, mse_E: {mse_loss_E:0.3f}, mse_TE: {mse_loss_TE:0.5f}')

        log_name = [datatype+i for i in ['epoch','mse_T', 'mse_E', 'mse_TE']]
        log_values = [epoch, mse_loss_T, mse_loss_E, mse_loss_TE]
        total_loss = mse_loss_T + mse_loss_E + mse_loss_TE
        return log_name, log_values, total_loss
    

    def save_results(model,Data,fname,Inds=Partitions,Edat=Edat):
        model.train_T = False
        model.train_E = False
        zT, zE, XrT, XrE = model((tf.constant(Data['T_dat']), tf.constant(Data[Edat])))
        XrE_from_XT = model.decoder_E(zT, training=False)
        XrT_from_XE = model.decoder_T(zE, training=False)

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
        
        train_fn(model=model_TE, optimizer=optimizer, XT=XT, XE=XE)
        
        if (step+1) % n_steps_per_epoch == 0:
            #Update epoch count
            epoch = epoch+1
            
            #Collect training metrics
            model_TE((train_T_dat, train_E_dat))
            train_log_name, train_log_values, train_total_loss = report_losses(model=model_TE ,epoch=epoch, datatype='train_', verbose=True)
            
            #Collect validation metrics
            model_TE((val_T_dat, val_E_dat))
            val_log_name, val_log_values, val_total_loss = report_losses(model=model_TE, epoch=epoch, datatype='val_', verbose=True)
            
            with open(dir_pth['logs']+fileid+'.csv', "a") as logfile:
                writer = csv.writer(logfile, delimiter=',')
                #Write headers to the log file
                if epoch == 1:
                    writer.writerow(train_log_name+val_log_name)
                writer.writerow(train_log_values+val_log_values)

            #Save networks conditionally:
            if (epoch > 1000) & (best_loss > val_total_loss):
                best_loss = val_total_loss
                save_fname = dir_pth['result']+fileid+'_best_loss'
                model_TE.save_weights(f'{save_fname}-weights.h5')
                save_results(model=model_TE, Data=D.copy(), fname=f'{save_fname}-summary.mat')
                print(f'Model saved with validation loss: {best_loss}')
            
    #Save model weights on exit
    save_fname = dir_pth['result']+fileid+'_exit'
    model_TE.save_weights(f'{save_fname}-weights.h5')
    save_results(model=model_TE, Data=D.copy(), fname=f'{save_fname}-summary.mat')

    print('\n\n--- fine tuning loop begins ---')
    #Fine tuning loop ----------------------------------------------------------------------
    #Each batch is now the whole training set
    for epoch in range(n_finetuning_steps):
        train_fn(model=model_TE, optimizer=optimizer, XT=train_T_dat, XE=train_E_dat)

        #Collect training metrics
        model_TE((train_T_dat, train_E_dat))
        train_log_name, train_log_values, train_total_loss = report_losses(model=model_TE ,epoch=epoch, datatype='train_', verbose=True)
        
        #Collect validation metrics
        model_TE((val_T_dat, val_E_dat))
        val_log_name, val_log_values, val_total_loss = report_losses(model=model_TE, epoch=epoch, datatype='val_', verbose=True)

        with open(dir_pth['logs']+fileid+'_'+str(n_finetuning_steps)+'_ft.csv', "a") as logfile:
            writer = csv.writer(logfile, delimiter=',')
            #Write headers to the log file
            if epoch == 0:
                writer.writerow(train_log_name+val_log_name)
            writer.writerow(train_log_values+val_log_values)


    save_fname = dir_pth['result']+fileid+f'_exit_{n_finetuning_steps}_ft'
    model_TE.save_weights(f'{save_fname}-weights.h5')
    save_results(model=model_TE, Data=D.copy(), fname=f'{save_fname}-summary.mat')
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

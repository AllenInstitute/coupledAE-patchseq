import argparse
import tensorflow as tf
from ae_model_def import Model_E_classifier
from data_funcs import TE_get_splits_45
import scipy.io as sio
import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer

parser = argparse.ArgumentParser()

parser.add_argument("--batchsize",         default=200,                     type=int,     help="Batch size")
parser.add_argument("--cvfold",            default=0,                       type=int,     help="45 fold CV sets (range from 0 to 44)")
parser.add_argument("--Edat",              default='pcipfx',                type=str,     help="`pc` or `ipfx` or `pcifpx`")
parser.add_argument("--latent_dim",        default=3,                       type=int,     help="Number of latent dims")
parser.add_argument("--n_epochs",          default=1000,                    type=int,     help="Number of epochs to train")
parser.add_argument("--n_steps_per_epoch", default=500,                     type=int,     help="Number of model updates per epoch")
parser.add_argument("--ckpt_save_freq",    default=500,                     type=int,     help="Frequency of checkpoint saves")
parser.add_argument("--run_iter",          default=0,                       type=int,     help="Run-specific id")
parser.add_argument("--model_id",          default='v1',                    type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='E_classifier',          type=str,     help="Experiment set")


def set_paths(exp_name='TEMP'):
    from pathlib import Path
    dir_pth = {}
    curr_path = str(Path().absolute())
    base_path=None
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
        dir_pth['data'] = base_path + 'dat/raw/patchseq-v4/'

    dir_pth['result'] = base_path + 'dat/result/' + exp_name + '/'
    dir_pth['checkpoint'] = dir_pth['result'] + 'checkpoints/'
    dir_pth['logs'] = dir_pth['result'] + 'logs/'

    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True)
    Path(dir_pth['checkpoint']).mkdir(parents=True, exist_ok=True)
    return dir_pth


class Datagen():
    """Iterator class to sample the dataset. Tensors T_dat and E_dat are provided at runtime.
    """

    def __init__(self, maxsteps, batchsize, E_dat, E_cat):
        self.E_cat = E_cat
        self.E_dat = E_dat
        self.batchsize = batchsize
        self.maxsteps = maxsteps
        self.n_samples = self.E_dat.shape[0]
        self.count = 0
        print('Initializing generator...')
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.maxsteps:
            self.count = self.count+1
            ind = np.random.randint(0, self.n_samples, self.batchsize)
            return (tf.constant(self.E_dat[ind, :], dtype=tf.float32),
                    tf.constant(self.E_cat[ind, :], dtype=tf.float32))
        else:
            raise StopIteration

def main(batchsize=200, cvfold=0, Edat = 'pcifpx',
         latent_dim=3,n_epochs=1000, n_steps_per_epoch=500, ckpt_save_freq=500,
         run_iter=0, model_id='E_class_v1', exp_name='E_classifier'):
    
    dir_pth = set_paths(exp_name=exp_name)

    fileid = model_id + \
        '_bs_' + str(batchsize) + \
        '_se_' + str(n_steps_per_epoch) + \
        '_ne_' + str(n_epochs) + \
        '_ld_' + str(latent_dim) + \
        '_cv_' + str(cvfold) + \
        '_ri_' + str(run_iter)
    fileid = fileid.replace('.', '-')

    if Edat == 'pcipfx':
        Edat = 'E_pcipxf'

    #Data operations and definitions:
    D = sio.loadmat(dir_pth['data']+'PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat',squeeze_me=True)
    D['E_pcipxf'] = np.concatenate([D['E_pc_scaled'],D['E_feature']],axis = 1)
    train_ind,val_ind,test_ind = TE_get_splits_45(matdict=D,cvfold=cvfold)

    Partitions = {'train_ind':train_ind,'val_ind':val_ind,'test_ind':test_ind}

    
    sorted_labels = sorted(list(set(zip(D['cluster'],D['cluster_id']))), key=lambda x: x[1])
    sorted_labels = [l[0] for l in sorted_labels]
    mlb = MultiLabelBinarizer(classes=sorted_labels)
    D['E_cat'] = mlb.fit_transform([{x} for x in D['cluster']])
    pred = mlb.inverse_transform(D['E_cat'])
    pred = np.array([x[0] for x in pred])
    assert np.array_equal(pred,D['cluster']), 'Binarizer not working as expected'


    train_E_dat = D[Edat][train_ind,:]
    train_E_cat = D['E_cat'][train_ind,:].astype(float)

    val_E_dat = D[Edat][val_ind,:]
    val_E_cat = D['E_cat'][val_ind,:].astype(float)

        
    Edat_var = np.nanvar(D[Edat],axis=0)
    maxsteps = tf.constant(n_epochs*n_steps_per_epoch)
    batchsize = tf.constant(batchsize)
        
    #Model definition
    optimizer_main = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_generator = tf.data.Dataset.from_generator(Datagen,output_types=(tf.float32, tf.float32),
                                                    args=(maxsteps,batchsize,train_E_dat,train_E_cat))

    model = Model_E_classifier(E_output_dim=train_E_dat.shape[1],
                            E_intermediate_dim=40,
                            E_gauss_noise_wt=1.0,
                            E_gnoise_sd=0.05,
                            E_dropout=0.1,
                            latent_dim=latent_dim,
                            n_labels=train_E_cat.shape[1])

    #Model training functions 
    @tf.function
    def train_fn(model, optimizer, XE, cE, train_E=False):
        """Enclose this with tf.function to create a fast training step. Function can be used for inference as well. 
        Arguments:
            cE: E category for training or validation
            XE: E data for training or validation
            train_T: {bool} -- Switch augmentation for T data on or off
            train_E {bool} -- Switch augmentation for E data on or off
            subnetwork {str} -- 'all' or 'E'. 'all' trains the full network, 'E' trains only the E arm.
        """

        with tf.GradientTape() as tape:
            zE,_ = model((XE, cE),train_E=train_E)
            trainable_weights = [weight for weight in model.trainable_weights]
            loss = sum(model.losses)

        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        return zE

    #Model logging functions
    def report_losses(model, epoch, datatype='train', acc = 0, verbose=False):
        ce_loss = model.ce_loss.numpy()
        if verbose:
            print(f'Epoch:{epoch:5d}, ce_loss: {ce_loss:0.3f}, acc: {acc:0.3f}')

        log_name = [datatype+i for i in ['epoch','ce_loss','acc']]
        log_values = [epoch, ce_loss, acc]
        return log_name, log_values

    def save_results(this_model,Data,fname,Inds=Partitions,Edat=Edat,mlb=mlb):
        all_E_dat = tf.constant(D[Edat],dtype=tf.float32)
        all_E_cat = tf.constant(D['E_cat'],dtype=tf.float32)
        zE, oH = model((all_E_dat, all_E_cat), training=False)
        oH = oH.numpy()
        bin_oH = (oH == np.max(oH,axis=1,keepdims=True)).astype(int)
        pred_E_cat = mlb.inverse_transform(bin_oH)
        pred_E_cat = np.array([x[0] for x in pred_E_cat])
        acc = (np.sum(pred_E_cat==D['cluster'])/D['cluster'].size)*100
        savemat = {'zE':zE.numpy(), 'pred_E_cat':pred_E_cat}
        savemat.update(Inds)
        sio.savemat(fname, savemat, do_compression=True)
        return

    #Main training loop ----------------------------------------------------------------------
    epoch=0
    best_val_acc = 0.0

    for step, (XE, cE) in enumerate(train_generator): 
        train_fn(model=model, optimizer=optimizer_main, XE=XE, cE=cE, train_E=True)

        if (step+1) % n_steps_per_epoch == 0:
            #Update epoch count
            epoch = epoch+1

            #Collect training metrics
            _,train_pred = model((train_E_dat, train_E_cat), train_E=False)
            train_pred = train_pred.numpy()
            train_pred = (train_pred == np.max(train_pred,axis=1,keepdims=True)).astype(int)
            train_acc = (np.sum(np.multiply(train_pred,train_E_cat))/train_pred.shape[0])
            train_log_name, train_log_values = report_losses(model=model ,epoch=epoch,  acc=train_acc, datatype='train_', verbose=True)

            #Collect validation metrics
            _, val_pred = model((val_E_dat, val_E_cat), train_E=False)

            val_pred = val_pred.numpy()
            val_pred = (val_pred == np.max(val_pred,axis=1,keepdims=True)).astype(int)
            val_acc = (np.sum(np.multiply(val_pred,val_E_cat))/val_pred.shape[0])
            val_log_name, val_log_values = report_losses(model=model, epoch=epoch, acc=val_acc, datatype='val_', verbose=True)

            with open(dir_pth['logs']+fileid+'.csv', "a") as logfile:
                writer = csv.writer(logfile, delimiter=',')
                #Write headers to the log file
                if epoch == 1:
                    writer.writerow(train_log_name+val_log_name)
                writer.writerow(train_log_values+val_log_values)

            if epoch % ckpt_save_freq == 0:
                #Save model weights
                model.save_weights(dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-weights.h5')
                #Save reconstructions and results for the full dataset:
                save_results(this_model=model,Data=D,fname=dir_pth['checkpoint']+fileid+'_ckptep_'+str(epoch)+'-summary.mat')
                
            if (val_acc>best_val_acc) and (epoch>20):
                #Save best accuracy model weights
                model.save_weights(dir_pth['result']+fileid+'-weights.h5')

                #Save reconstructions and results for the full dataset:
                save_results(this_model=model,Data=D,fname=dir_pth['result']+fileid+'-summary.mat')
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

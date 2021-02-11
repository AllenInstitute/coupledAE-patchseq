#!/bin/bash
#PBS -q celltypes
#PBS -N TE_CS1cv3ld1aug
#PBS -m a
#PBS -r n
#PBS -l ncpus=8
#PBS -l mem=16g,walltime=4:00:00
#PBS -o /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/logs/TE_CS1cv3ld1aug.out
#PBS -j oe
cd /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/code/Patchseq-bioarxiv/
source activate tf21-cpu
python -m ae_model_train_v3  --batchsize 200  --cvfold 1  --alpha_T 1.0  --alpha_E 1.0  --lambda_TE 1.0  --augment_decoders 1  --latent_dim 3  --n_epochs 1500  --n_steps_per_epoch 500  --n_finetuning_steps 500  --run_iter 0  --model_id CS  --exp_name TE_CS

csTE=1.0
aE=1.0
for aM in 0.0 1.0 10.0 100.0
do
    for cv in {0..8}
    do
        jobid="TE_aM_"$aM"_cv_"$cv
        echo '#!/bin/bash'>subjob.bash
        echo '#PBS -q celltypes'>>subjob.bash
        echo '#PBS -N '${jobid//./-} >>subjob.bash
        echo '#PBS -m a'>>subjob.bash
        echo '#PBS -r n'>>subjob.bash
        echo '#PBS -l ncpus=8'>>subjob.bash
        echo '#PBS -l mem=16g,walltime=10:00:00'>>subjob.bash
        echo '#PBS -o /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/logs/'${jobid//./-}'.out'>>subjob.bash
        echo '#PBS -j oe'>>subjob.bash
        echo 'cd /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/code/Patchseq-AE-Bioarxiv/'>>subjob.bash
        echo 'source activate tf20-cpu'>>subjob.bash
        echo 'python -m ae_model_train --batchsize 200 --cvfold '$cv' --alpha_T 1.0 --alpha_E '$aE' --alpha_M '$aM' --lambda_TE '$csTE' --latent_dim 3 --n_epochs 1500 --n_steps_per_epoch 500 --ckpt_save_freq 2000 --n_finetuning_steps 500 --run_iter 0 --model_id v3 --exp_name TE_Patchseq_Bioarxiv'>>subjob.bash
        echo '...'
        sleep 1
        wait
        #qsub subjob.bash
        echo 'Job: '${jobid//./-}' '
        cat subjob.bash
    done
done

rm subjob.bash

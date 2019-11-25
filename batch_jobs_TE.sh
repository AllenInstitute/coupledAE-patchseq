for csTE in 0 0.1 1.0 10.0
do
    for aE in 0.1 0.2 0.5 1.0
    do
        for cv in {0..8}
        do
            jobid="TE_"$cv"cv"$csTE"csTE"$aE"aE"
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
            echo 'python -m ae_model_train --batchsize 200 --cvfold '$cv' --alpha_T 1.0 --alpha_E '$aE' --alpha_M '$aE' --lambda_TE '$csTE' --latent_dim 3 --n_epochs 1500 --n_steps_per_epoch 500 --run_iter 0 --model_id v1 --exp_name TE_Patchseq_Bioarxiv'>>subjob.bash
            echo '...'
            sleep 1
            wait
            #qsub subjob.bash
            echo 'Job: '${jobid//./-}' '
            cat subjob.bash
        done
    done
done

rm subjob.bash

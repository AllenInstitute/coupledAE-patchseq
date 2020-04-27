for cv in {0..44}
do
    for csTE in 0 0.5 1.0 2.0 5.0
    do
        for aug in 0
        do
            jobid="TE_NM"$cv"cv"$csTE"csTE"$aug"aug"
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
            echo 'source activate tf21-cpu'>>subjob.bash
            echo 'python -m ae_model_train_v2' \
                    ' --batchsize 200' \
                    ' --cvfold '$cv \
                    ' --alpha_T 1.0'\
                    ' --alpha_E 1.0'\
                    ' --lambda_TE '$csTE \
                    ' --augment_decoders '$aug \
                    ' --latent_dim 3'\
                    ' --n_epochs 1500'\
                    ' --n_steps_per_epoch 500'\
                    ' --ckpt_save_freq 500'\
                    ' --n_finetuning_steps 100'\
                    ' --run_iter 0'\
                    ' --model_id NM'\
                    ' --exp_name TE_NM'>>subjob.bash
            echo '...'
            sleep 1
            wait
            qsub subjob.bash
            echo 'Job: '${jobid//./-}' '
            cat subjob.bash
        done
    done
done

rm subjob.bash

ri=0

for ld in {3,5,10}
do
    for csTE in 1.0
    do
        for cv in {0..19}
        do
            for aug in {0,1}
            do
                jobid="TE_CS"$cv"cv"$ld"ld"$aug"aug"
                echo '#!/bin/bash'>subjob.bash
                echo '#PBS -q celltypes'>>subjob.bash
                echo '#PBS -N '${jobid//./-} >>subjob.bash
                echo '#PBS -m a'>>subjob.bash
                echo '#PBS -r n'>>subjob.bash
                echo '#PBS -l ncpus=8'>>subjob.bash
                echo '#PBS -l mem=16g,walltime=4:00:00'>>subjob.bash
                echo '#PBS -o /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/logs/'${jobid//./-}'.out'>>subjob.bash
                echo '#PBS -j oe'>>subjob.bash
                echo 'cd /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/code/Patchseq-bioarxiv/'>>subjob.bash
                echo 'source activate tf21-cpu'>>subjob.bash
                echo 'python -m ae_model_train_v3' \
                        ' --batchsize 200' \
                        ' --cvfold '$cv \
                        ' --alpha_T 1.0'\
                        ' --alpha_E 1.0'\
                        ' --lambda_TE '$csTE \
                        ' --augment_decoders '$aug \
                        ' --latent_dim '$ld \
                        ' --n_epochs 1500'\
                        ' --n_steps_per_epoch 500'\
                        ' --n_finetuning_steps 500'\
                        ' --run_iter '$ri \
                        ' --model_id CS'\
                        ' --exp_name TE_CS'>>subjob.bash
                echo '...'
                sleep 0.1
                wait
                qsub subjob.bash
                echo 'Job: '${jobid//./-}' '
                #cat subjob.bash
            done
        done
    done
done

rm subjob.bash
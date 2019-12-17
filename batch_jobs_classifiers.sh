for cv in {0..8}
do 
    for csTE in 1.0
    do
        for aE in 1.0
        do
            for start_i in {0..6200..200}
            #for start_i in {0..3000..200}
            do
                rn="n88"
                #rn="n60"
                stop_i=$((start_i + 200))

                jobid="LRC_bal_"$cv"cv"$csTE"csTE"$aE"aE_"$start_i"-"$stop_i
                echo '#!/bin/bash'>subjob.bash
                echo '#PBS -q celltypes'>>subjob.bash
                echo '#PBS -N '${jobid//./-} >>subjob.bash
                echo '#PBS -m a'>>subjob.bash
                echo '#PBS -r n'>>subjob.bash
                echo '#PBS -l ncpus=4'>>subjob.bash
                echo '#PBS -l mem=2g,walltime=5:00:00'>>subjob.bash
                echo '#PBS -o /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/logs/'${jobid//./-}'.out'>>subjob.bash
                echo '#PBS -j oe'>>subjob.bash
                echo 'cd /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/code/Patchseq-AE-Bioarxiv/'>>subjob.bash
                echo 'source activate tf20-cpu'>>subjob.bash

                echo 'python -m analysis_parallel_classifier --cvfold '$cv' --alpha_T 1.0 --alpha_E '$aE' --lambda_TE '$csTE' --root_node '$rn' --start_i '$start_i' --stop_i '$stop_i' --embedding zE --rand_seed 0 --exp_name LR_v2_bal'>>subjob.bash
                echo '...'
                sleep 1
                wait
                qsub subjob.bash
                echo 'Job: '${jobid//./-}' '
                cat subjob.bash
            done
        done
    done
done
rm subjob.bash

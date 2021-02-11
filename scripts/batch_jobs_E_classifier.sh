for ld in 3 5 10
do
    for cv in {0..10}
    do
        jobid="E_ld_"$ld"_cv_"$cv
        echo '#!/bin/bash'>subjob.bash
        echo '#PBS -q celltypes'>>subjob.bash
        echo '#PBS -N '${jobid//./-} >>subjob.bash
        echo '#PBS -m a'>>subjob.bash
        echo '#PBS -r n'>>subjob.bash
        echo '#PBS -l ncpus=8'>>subjob.bash
        echo '#PBS -l mem=16g,walltime=2:30:00'>>subjob.bash
        echo '#PBS -o /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/logs/'${jobid//./-}'.out'>>subjob.bash
        echo '#PBS -j oe'>>subjob.bash
        echo 'cd /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/code/Patchseq-bioarxiv/'>>subjob.bash
        echo 'source activate tf21-cpu'>>subjob.bash
        echo 'python -m ae_model_trainclassifier --cvfold '$cv' --fiton ttype33     --stratify_by consensus33 --latent_dim '$ld' --model_id v2 --exp_name E_classifiers_v2'>>subjob.bash
        echo 'python -m ae_model_trainclassifier --cvfold '$cv' --fiton consensus33 --stratify_by ttype33     --latent_dim '$ld' --model_id v2 --exp_name E_classifiers_v2'>>subjob.bash
        echo '...'
        sleep 1
        wait
        qsub subjob.bash
        echo 'Job: '${jobid//./-}' '
        cat subjob.bash
    done
done

rm subjob.bash
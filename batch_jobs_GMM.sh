cv=0
for ri in {10..20}
do
    for minn in {10..45..1}
    do
        jobid="GMMfit_cv_"$cv"_n_"$minn
        echo '#!/bin/bash'>subjob.bash
        echo '#PBS -q celltypes'>>subjob.bash
        echo '#PBS -N '${jobid//./-} >>subjob.bash
        echo '#PBS -m a'>>subjob.bash
        echo '#PBS -r n'>>subjob.bash
        echo '#PBS -l ncpus=4'>>subjob.bash
        echo '#PBS -l mem=2g,walltime=0:45:00'>>subjob.bash
        echo '#PBS -o /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/logs/'${jobid//./-}'.out'>>subjob.bash
        echo '#PBS -j oe'>>subjob.bash

        echo 'cd /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/code/Patchseq-bioarxiv/'>>subjob.bash
        echo 'source activate tf21-cpu'>>subjob.bash
        echo 'python -m analysis_denovo_script' \
                        ' --representation_pth TE_NM' \
                        ' --exp_name gmm_model_select_cv_0' \
                        ' --cvfold '$cv \
                        ' --alpha_T 1.0'\
                        ' --alpha_E 1.0'\
                        ' --lambda_TE 1.0'\
                        ' --augmented_decoders 1'\
                        ' --run_iter '$ri \
                        ' --perc 100'\
                        ' --min_component '$minn >>subjob.bash
        echo '...'
        sleep 0.2
        wait
        qsub subjob.bash
        echo 'Job: '${jobid//./-}' '
        #cat subjob.bash
    done
done
rm subjob.bash
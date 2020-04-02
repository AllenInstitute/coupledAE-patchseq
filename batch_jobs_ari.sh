for cv in {0..44}
do
    
    jobid="ARIcalc_cv_"$cv
    echo '#!/bin/bash'>subjob.bash
    echo '#PBS -q celltypes'>>subjob.bash
    echo '#PBS -N '${jobid//./-} >>subjob.bash
    echo '#PBS -m a'>>subjob.bash
    echo '#PBS -r n'>>subjob.bash
    echo '#PBS -l ncpus=4'>>subjob.bash
    echo '#PBS -l mem=2g,walltime=2:00:00'>>subjob.bash
    echo '#PBS -o /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/logs/'${jobid//./-}'.out'>>subjob.bash
    echo '#PBS -j oe'>>subjob.bash

    echo 'cd /allen/programs/celltypes/workgroups/mousecelltypes/Rohan/code/Patchseq-bioarxiv/'>>subjob.bash
    echo 'source activate tf21-cpu'>>subjob.bash
    echo 'python -m analysis_ref_hierarchy_conf' \
                    ' --cvfold '$cv >>subjob.bash
    echo '...'
    sleep 0.2
    wait
    qsub subjob.bash
    echo 'Job: '${jobid//./-}' '
    #cat subjob.bash
    done
rm subjob.bash
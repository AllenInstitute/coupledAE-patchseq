for cca_dim in {3,5,10}
do
    for pc_dim in {10,20,50}
    do
        for cv in {0..19}
        do      
            echo ccadim $cca_dim   pcdim $pc_dim   cv $cv
            python -m pc-cca_TE \
                     --cvfold $cv \
                     --pc_dim_T $pc_dim \
                     --pc_dim_E $pc_dim \
                     --cca_dim $cca_dim \
                     --model_id PCCCA\
                     --exp_name TE_CS
        done
    done
done
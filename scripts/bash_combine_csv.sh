#Combines csv files. Used to combine GMM fit files. 
result_file="gmmfit_combined_perc_100-0_fiton_zT_aT_1-0_aE_1-0_cs_1-0_ad_1_minn_10_maxn_60.csv"
touch $result_file
head -1 gmmfit_perc_100-0_ld_3_fiton_zT_aT_1-0_aE_1-0_cs_1-0_ad_1_cv_19minn_43_maxn_44.csv > $result_file
for filename in $(ls gmmfit_perc_100-0_ld_3_fiton_zT_aT_1-0_aE_1-0_cs_1-0_ad_1*.csv); do sed 1d $filename >> $result_file; done


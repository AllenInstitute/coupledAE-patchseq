**[Consistent cross-modal identification of cortical neurons with coupled autoencoders](https://www.biorxiv.org/content/10.1101/2020.06.30.181065v1)**

**Abstract**
>Consistent identification of neurons and neuronal cell types across different observation modalities is an important problem in neuroscience. Here, we present an optimization framework to learn coordinated representations of multimodal data, and apply it to a large Patch-seq dataset of mouse cortical interneurons. Our approach reveals strong alignment between transcriptomic and electrophysiological profiles of neurons, enables accurate cross-modal data prediction, and identifies cell types that are consistent across modalities.

**Data**
 - [Allen Institute Patch-seq dataset](https://portal.brain-map.org/explore/classes/multimodal-characterization)
 - [Processed data used as input for coupled autoencoders](https://www.dropbox.com/s/nmhd3wzw4re9ve7/PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat?dl=0)

`Caution: The repository is being refactored.`
 - `ae_model_def.py` has coupled autoencoder implementation used with the patchseq data.
 - `ae_model_train_v2.py` specifies the training.
 - `loaddataset_v2.ipynb` includes summary plots characterizing the dataset.
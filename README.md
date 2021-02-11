**[Consistent cross-modal identification of cortical neurons with coupled autoencoders](https://www.biorxiv.org/content/10.1101/2020.06.30.181065v2)**

**Abstract**
>Consistent identification of neurons in different experimental modalities is a key problem in neuroscience. While methods to perform multimodal measurements in the same set of single neurons have become available, parsing complex relationships across different modalities to uncover neuronal identity is a growing challenge. Here, we present an optimization framework to learn coordinated representations of multimodal data, and apply it to a large multimodal dataset profiling mouse cortical interneurons. Our approach reveals strong alignment between transcriptomic and electrophysiological characterizations, enables accurate cross-modal data prediction, and identifies cell types that are consistent across modalities.

**Data**
 - [Allen Institute Patch-seq dataset](https://portal.brain-map.org/explore/classes/multimodal-characterization)
 - [Processed data used as input for coupled autoencoders](https://www.dropbox.com/s/nmhd3wzw4re9ve7/PS_v5_beta_0-4_pc_scaled_ipxf_eqTE.mat?dl=0)

An executable, minimal version of this repository is hosted on [CodeOcean](https://codeocean.com/capsule/6320801/tree/v1)

`Caution: The repository is being refactored.`

 - `/refactor` is a minimal implementation of the coupled autoencoder (`/refactor/README.md`)
 - `ae_model_def_v3.py` has coupled autoencoder implementation used with the patchseq data.
 - `ae_model_train_v3.py` specifies the training.
 - `loaddataset_v2.ipynb` includes summary plots characterizing the dataset.
 - `analysis_denovo_clustering.ipynb` Unsupervised clustering results + consensus cluster analysis.
 - `analysis_reconstructions.ipynb` Within- and cross-modality reconstruction results.
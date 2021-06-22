**[Consistent cross-modal identification of cortical neurons with coupled autoencoders](https://www.nature.com/articles/s43588-021-00030-1)**


```bibtex
@article{gala2021consistent,
  title={Consistent cross-modal identification of cortical neurons with coupled autoencoders},
  author={Gala, Rohan and Budzillo, Agata and Baftizadeh, Fahimeh and Miller, Jeremy and Gouwens, Nathan and Arkhipov, Anton and Murphy, Gabe and Tasic, Bosiljka and Zeng, Hongkui and Hawrylycz, Michael and S{\"u}mb{\"u}l, Uygar},
  journal={Nature Computational Science},
  volume={1},
  number={2},
  pages={120--127},
  year={2021},
  publisher={Nature Publishing Group}
}
```

### Abstract

Consistent identification of neurons in different experimental modalities is a key problem in neuroscience. While methods to perform multimodal measurements in the same set of single neurons have become available, parsing complex relationships across different modalities to uncover neuronal identity is a growing challenge. Here, we present an optimization framework to learn coordinated representations of multimodal data, and apply it to a large multimodal dataset profiling mouse cortical interneurons. Our approach reveals strong alignment between transcriptomic and electrophysiological characterizations, enables accurate cross-modal data prediction, and identifies cell types that are consistent across modalities.

### Data

 - [Allen Institute Patch-seq data browser](https://knowledge.brain-map.org/data/1HEYEW7GMUKWIQW37BO/specimens)
 - `data/proc/` contains the processed dataset used for Gala et al. 2021.
 - see `notebooks/data_proc_T.ipynb` and `notebooks/data_proc_E.ipynb` for pre-processing steps.

### Code

 - create a `conda` environment, and install depencies (see `requirements.yml`). The models can be run with any `tensorflow ` versions `2.1` to `2.5`
 - clone this repository.
 - navigate to the location with `setup.py` in this reposiory, and use `pip install -e .`
 - use `cplAE_TE/train.py` to start training a model

⚠️`Caution`: This repository is being refactored. Please contact corresponding authors for specific questions about code in this repository. 

You can also play around with a minimal version of the coupled autoencoders code (see `minimal` folder in this repository) hosted on a cloud environment at [CodeOcean](https://codeocean.com/capsule/6320801).


### See also:

[A coupled autoencoder approach for multi-modal analysis of cell types, Gala R. et al, Advances in Neural Information Processing Systems 32, 9267--9276, 2019](https://proceedings.neurips.cc/paper/2019/hash/30d4e6422cd65c7913bc9ce62e078b79-Abstract.html). 

The main points covered by earlier work:
 - We described the problem of collapsing representations encountered when maximizing correlation between representations of coupled autoencoders. 
 - We showed that our solution is an efficient way to effectively _whiten_ the representations.
 - We used this model to relate transcriptomic and physiological profiles obtained with patch-seq technology.

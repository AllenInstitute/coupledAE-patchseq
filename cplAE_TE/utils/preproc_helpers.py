import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import json
from cplAE_TE.utils.load_helpers import get_paths
    
def get_spc_names_v2():
    """Parse sparse pc names from .json generated by Nathan, for revised dataset in Gouwens et al. 2020. 

    Returns:
        E_name_array: np.array of object type containing names of Ephys features
    """


    path = get_paths()
    with open(path['v2_spc_names']) as f:
        new_comp_info = json.load(f)
    new_col_lookup = {}
    counter = 0
    for ci in new_comp_info:
        new_col_lookup[ci["key"]] = [str(counter + i)
                                     for i, v in enumerate(ci["indices"])]
        counter += len(ci["indices"])

    keys_ = []
    columns_ = []
    for key, value in new_col_lookup.items():
        for i, v in enumerate(value):
            keys_.append(key+'_'+str(i))
            columns_.append(v)
    keys_ = np.array(keys_)
    columns_ = np.array([int(c) for c in columns_])

    sort_ind = np.argsort(columns_)
    keys_ = keys_[sort_ind]
    columns_ = columns_[sort_ind]
    keys_ = [k.strip() for k in keys_]
    E_name_array = np.array(keys_, dtype=object)
    return E_name_array


def standardize_E(df, feature_list, z_absmax=6.0, plots=False, verbose=True):
    """Remove outliers and standardize columns of the (Ephys.) dataframe (either principle component scores or IPFX features)

    Args:
        df: E_pc or E_ipfx
        feature_list: list of features to standardize in the dataframe
        z_absmax: Number of standard deviations away from mean to fix values of outliers. Choose conservatively.
        plots (bool): plot distribution of values after z-scoring and outlier removal
        verbose (bool): print number number of outliers

    Returns:
        df: Dataframe with z-scored and outlier-removed values
        feature_means
    """

    feature_mean = []
    feature_std = []
    for i, feature in enumerate(feature_list):
        e = df[feature].copy()
        low = e.dropna().quantile(q=0.01)
        high = e.dropna().quantile(q=0.99)

        # ignore nans, and outliers defined by percentiles to calculate means and std
        values = e[e.between(low, high)]
        e_mean = np.mean(values)
        e_std = np.std(values)
        e = (e-e_mean)/e_std
        feature_mean.append(e_mean)
        feature_std.append(e_std)

        # set outliers to boundary
        num_set = (e.gt(z_absmax) | e.lt(-z_absmax)).sum()
        nan_ind = e.isna()
        e[e > z_absmax] = z_absmax
        e[e < -z_absmax] = -z_absmax
        e[nan_ind] = np.nan
        df[feature] = e

        if (num_set > 0) & verbose:
            print('{} outliers in {}'.format(num_set, feature))

        if plots:
            fig_ncols = 5
            # each figure is a single row with `fig_ncols` columns
            if i % fig_ncols == 0:
                plt.figure(figsize=(17, 3))
            plt.subplot(1, fig_ncols, i % fig_ncols+1)
            plt.hist(e[~nan_ind])
            plt.title(f'{feature:s}, \n {e.isna().sum():d}')

            if (i+1) % fig_ncols == 0:
                plt.tight_layout()
                plt.show()

    return df, feature_mean, feature_std


def scale(X, center=False):
    """Scale the values such that total variance for this sub-experiment = number of PC retained
    """
    # Remove mean
    if center:
        X = (X-np.mean(X, axis=0))
    n_retained_pc = X.shape[1]
    scale_factor = (n_retained_pc**0.5)/(np.sum(np.var(X, axis=0))**0.5)
    Xscaled = X*scale_factor
    return Xscaled


def scale_E_scores(L):
    """Uses the principle component vectors and loadings to reconstruct the original time-series for sub-experiments. 

    Args:
        L: pc summary .pkl file with original principle component vectors and loadings

    Returns:
        E_df: dataframe with scaled principle component scores
        TimeSeries: dict with [cells x time] reconstructed time series from the pcs for each sub-experiment
        Scores: dict with [cells x loadings] that are not normalized for each sub-expt
        TimeSeries_scaled: dict with [cells x time] reconstructed time series from the scaled values of scores for each sub-experiment
        Scores: dict with [cells x loadings] that are scaled by the number of principle components used to describe that sub-experiment
    """
    
    # dict with number of components retained for each sub-experiment. e.g
    # 'first_ap_v': 5 indicates that the first_ap_v_0, first_ap_v_1, ... , first_ap_v_4 are present in the list of features. 
    retained_pc = {'first_ap_v': 5,
             'first_ap_dv':6,
             'isi_shape':3,
             'step_subthresh':2,
             'subthresh_norm':4,
             'inst_freq':6,
             'spiking_upstroke_downstroke_ratio':2,
             'spiking_peak_v':2,
             'spiking_fast_trough_v':2,
             'spiking_threshold_v':3,
             'spiking_width':2,
             'inst_freq_norm':7}
    
    sub_expt_names = list(retained_pc.keys())

    Scores = {}
    TimeSeries = {}
    Scores_scaled = {}
    TimeSeries_scaled = {}

    for sub_expt in sub_expt_names:
        inds = np.arange(retained_pc[sub_expt])
        Vec = L[sub_expt]['loadings'][:, inds].transpose()

        # reconstruct time-series from original scores
        Scores[sub_expt] = L[sub_expt]['transformed_values'][:, inds]
        TimeSeries[sub_expt] = Scores[sub_expt] @ Vec
        
        # reconstruct time-series from scaled scores
        Scores_scaled[sub_expt] = scale(X=Scores[sub_expt])
        TimeSeries_scaled[sub_expt] = Scores_scaled[sub_expt] @ Vec

    # plot the variances of scaled scores
    plt.figure(figsize=(15, 4))
    x_shift = 0
    for sub_expt in sub_expt_names:
        x = np.arange(Scores_scaled[sub_expt].shape[1]) + x_shift
        y = Scores_scaled[sub_expt].var(axis=0)
        plt.plot(x, y, '.-', label=f'{sub_expt} {np.sum(y):0.2f}')
        x_shift = x_shift + Scores_scaled[sub_expt].shape[1]

    ax = plt.gca()
    ax.set_xlim(-1, 60)
    ax.set_ylabel('~std')
    plt.legend()
    plt.show()

    # concatenate scores into dataframe:
    df = {}
    for sub_expt in sub_expt_names:
        for i in range(retained_pc[sub_expt]):
            df[f'{sub_expt}_{i:d}'] = Scores_scaled[sub_expt][:, i]

    E_df = pd.DataFrame.from_dict(df)
    return E_df, TimeSeries, TimeSeries_scaled

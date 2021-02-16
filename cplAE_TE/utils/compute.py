import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.metrics import silhouette_samples


class CCA_extended(CCA):
  
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        return
      
    def inverse_transform_xy(self,X,Y):
        """
        This module calculates the inverse transform for both X and Y
        CCA.inverse_transform module only calculates reconstructions for X
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        Y = check_array(Y, dtype=FLOAT_DTYPES)
        
        x = np.matmul(X, self.x_loadings_.T)
        x *= self.x_std_
        x += self.x_mean_
        
        y = np.matmul(Y, self.y_loadings_.T)
        y *= self.y_std_
        y += self.y_mean_
        return x,y
    
    
def contingency(a, b, unique_a, unique_b):
    """Populate contingency matrix. Rows and columns are not normalized in any way.
    
    Args:
        a (np.array): labels
        b (np.array): labels
        unique_a (np.array): unique list of labels. Can have more entries than np.unique(a)
        unique_b (np.array): unique list of labels. Can have more entries than np.unique(b)

    Returns:
        C (np.array): contingency matrix.
    """
    assert a.shape == b.shape
    C = np.zeros((np.size(unique_a), np.size(unique_b)))
    for i, la in enumerate(unique_a):
        for j, lb in enumerate(unique_b):
            C[i, j] = np.sum(np.logical_and(a == la, b == lb))
    return C


def contingency_metric(X, verbose=True, factor=1):
    """Calculate measures to analyze contingency matrix.

    Args:
        X (np.array): contingency matrix
        verbose (bool): print measures

    Returns:
        reliable_row_frac: rows for which the diagonal entry >= row maximum > 0
        reliable_col_frac: cols for which the diagonal entry >= row maximum > 0
        reliable_diag_frac: reliable_row_frac & reliable_col_frac
        bad_diag
        f_consistency
        f_occupancy
    """
    assert X.shape[0] == X.shape[1], 'X should be a square matrix'

    row_m = []
    col_m = []
    for ii in range(X.shape[0]):
        row = X[ii, :].copy()
        row[ii] = 0
        row_m.append(np.logical_and(np.all(X[ii, ii] >= factor*row), X[ii, ii] > 0))

        col = X[:, ii].copy()
        col[ii] = 0
        col_m.append(np.logical_and(np.all(X[ii, ii] >= factor*col), X[ii, ii] > 0))
    reliable_row_frac = np.mean(row_m)
    reliable_col_frac = np.mean(col_m)
    reliable_diag_frac = np.mean(np.logical_and(row_m, col_m))
    unreliable_clusters = np.flatnonzero(np.logical_or(~np.array(row_m), ~np.array(col_m)))

    bin_X = X > 0
    f_consistency = np.trace(X)/np.sum(X, axis=None)
    f_occupancy = np.trace(bin_X)/np.shape(bin_X)[0]

    if verbose:
        print(f'Total samples: {np.sum(X.astype(int)):d}')
        print(f'reliable cluster frac: {reliable_diag_frac:0.3f}')

    return reliable_row_frac, reliable_col_frac, reliable_diag_frac, unreliable_clusters, f_consistency, f_occupancy


def silhouette_analysis(z, lbl):
    """returns dataframes with sample- and label-level silhouette scores.

    Args:
        z (np.ndarray): positions
        lbl (np.ndarray): cluster labels

    Returns:
        df: sample level silhouette scores
        df: label level silhouette scores
    """
    s = silhouette_samples(z, lbl, metric='euclidean')
    df = pd.DataFrame({'sil': s, 'lbl': lbl})
    df_lbl = df.groupby(by='lbl').agg([np.mean,np.std]).reset_index()
    df_lbl = df_lbl.sort_values(by=('sil','mean')).reset_index(drop=True)
    return df, df_lbl
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

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
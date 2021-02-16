from copy import deepcopy

import numpy as np
import pandas as pd
from cplAE_TE.utils.load_helpers import get_paths, load_dataset
from cplAE_TE.utils.tree_helpers import get_merged_ordered_classes
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


def supervised_classification(Fold, representation_id='zT', n_classes_list=np.arange(5, 61, 5), verbose=False):
    """Perform supervised classification with QDA

    Args:
        Fold (Dict): Fold summary, with training, validation, and test splits.
        representation_id (str) = 'zT' or 'zE'

    Returns:
        results_df: dataframe with results
    """

    n_min_samples = 6 #per-class number of samples used by QDA to fit.
    path = get_paths()
    O = load_dataset()

    results = {}
    results['n_components'] = []
    results['n_htree_classes'] = []
    results['acc_train'] = []
    results['acc_val'] = []
    results['acc_test'] = []
    results['acc_most_freq'] = []
    results['acc_prior'] = []

    dummy_most_freq = DummyClassifier(strategy="most_frequent")
    dummy_prior = DummyClassifier(strategy="stratified")

    for n_classes in n_classes_list:

        merged_labels, _ = get_merged_ordered_classes(data_labels=deepcopy(O['cluster']),
                                                      htree_file=path['htree'],
                                                      n_required_classes=n_classes,
                                                      verbose=False)

        X_train = deepcopy(Fold[representation_id][Fold['train_ind']])
        y_train = deepcopy(merged_labels[Fold['train_ind']])
        ind_train = np.arange(0, np.shape(X_train)[0])

        X_val = deepcopy(Fold[representation_id][Fold['val_ind']])
        y_val = deepcopy(merged_labels[Fold['val_ind']])
        ind_val = np.arange(0, np.shape(X_val)[0])

        X_test = deepcopy(Fold[representation_id][Fold['test_ind']])
        y_test = deepcopy(merged_labels[Fold['test_ind']])
        ind_test = np.arange(0, np.shape(X_test)[0])

        #Remove types with low sample counts in training set
        df = pd.DataFrame({'ind': ind_train, 'lbl': y_train})
        df_train = df[df.groupby('lbl')['lbl'].transform('count').ge(n_min_samples)]
        keep_ind = df_train['ind'].values
        X_train = X_train[keep_ind, :]
        y_train = y_train[keep_ind]

        #Print types that were ignored
        if verbose:
            df_train_del = df[df.groupby('lbl')['lbl'].transform('count').lt(n_min_samples)]
            print(df_train_del['lbl'].value_counts())
        
        #Remove types from validation set that are not represented in the training set
        df = pd.DataFrame({'ind':ind_val,'lbl':y_val})
        df_val = df[df['lbl'].isin(y_train)]
        keep_ind = df_val['ind'].values
        X_val = X_val[keep_ind,:]
        y_val = y_val[keep_ind]

        #Remove types from test set that are not represented in the training set
        df = pd.DataFrame({'ind':ind_test,'lbl':y_test})
        df_test = df[df['lbl'].isin(y_test)]
        keep_ind = df_test['ind'].values
        X_test = X_test[keep_ind,:]
        y_test = y_test[keep_ind]

        #QDA related metrics
        qda = QDA(reg_param=1e-2, store_covariance=True)
        qda.fit(X_train, y_train)
        y_train_pred = qda.predict(X_train)
        y_val_pred = qda.predict(X_val)
        y_test_pred = qda.predict(X_test)
        
        results['n_htree_classes'].append(n_classes)
        results['n_components'].append(np.unique(qda.classes_).size)
        results['acc_train'].append(accuracy_score(y_train, y_train_pred))
        results['acc_val'].append(accuracy_score(y_val, y_val_pred))
        results['acc_test'].append(accuracy_score(y_test, y_test_pred))
            
        #For dummy classifiers
        dummy_most_freq.fit(Fold[representation_id], merged_labels)
        most_freq_pred = dummy_most_freq.predict(Fold[representation_id])

        dummy_prior.fit(Fold[representation_id], merged_labels)
        prior_pred = dummy_prior.predict(Fold[representation_id])
        
        results['acc_most_freq'].append(accuracy_score(merged_labels, most_freq_pred))
        results['acc_prior'].append(accuracy_score(merged_labels, prior_pred))

    results_df = pd.DataFrame(results)
    return results_df

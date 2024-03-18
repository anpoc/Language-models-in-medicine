import numpy as np
from typing import Optional, Union
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    f1_score, precision_score, recall_score, top_k_accuracy_score, roc_auc_score


def calculate_statistics(vals:np.array) -> list:
    """Calculate the mean, standard deviation, minimum, Q1, Q2, Q3, and max (descriptive statistics)
        of an array.

    Args:
        vals (np.array): values.

    Returns:
        list: descriptive statistics.
    """
    if vals.dtype != np.float64: vals = vals.astype('float64')
    statistics = [np.nanmean(vals), np.nanstd(vals)]
    statistics += np.nanpercentile(vals, [0, 25, 50, 75, 100]).tolist()
    return statistics


def get_results_template(task:str) -> dict:
    """Get the template of the results.

    Args:
        task (str): approach or task from wich the results are produced.

    Returns:
        dict: template.
    """
    if task == 'decode':
        template = {
            'col_names': ['model', 'data', 'perplexity_mean', 'perplexity_std', 'perplexity_min', 
            'perplexity_25', 'perplexity_50', 'perplexity_75', 'perplexity_max', 'time'], 
            'values': []
        }
    else:
        template = {
            'col_names': ['model', 'data', 'accuracy', 'balanced_accuracy', 'accuracy_top3', 
            'accuracy_top5', 'auc', 'c_auc', 'f1', 'precision', 'recall', 'c_f1', 'c_precision', 
            'c_recall', 'time'], 
            'values': []
        }
    return template


def measure_classification(y_true:Union[list, np.ndarray], y_pred:Union[list, np.ndarray], 
    as_score:bool=True, multilabel:bool=False, exceptions:Optional[list]=None) -> list:
    """Measure the classifcation performance.

    Args:
        y_true (Union[list, np.ndarray]): true labels.
        y_pred (Union[list, np.ndarray]): predicted labels.
        as_score (bool, optional): whether the predicted labels are given as scores or not. 
            Defaults to True.
        multilabel (bool, optional): whether the classification is multilabel or not (i.e. multiclass). 
            Defaults to False.
        exceptions (Optional[list], optional): whether there are more than one possible label for 
            a multiclass classification. These are treated as exclusive. Defaults to None.

    Returns:
        list:  performances.
    """
    y_pred = np.array(y_pred) if type(y_pred) == list else y_pred

    if multilabel:
        return [None, None, None, None, roc_auc_score(y_true, y_pred), 
            roc_auc_score(y_true, y_pred, average=None), None, None, None, None, None, None]

    if as_score:
        _, l = y_pred.shape
        labels = list(range(l))
        y_score = y_pred[:, int(l == 2):]
        y_pred = np.argmax(y_pred, axis=-1)
        if not(exceptions is None):
            for pos, idx in enumerate(exceptions[0]):
                if y_pred[idx] in exceptions[1][pos]:
                    y_true[idx] = y_pred[idx]

    performance = [
        accuracy_score(y_true, y_pred),
        balanced_accuracy_score(y_true, y_pred),
        top_k_accuracy_score(y_true, y_score, k=3, labels=labels) if as_score else None,
        top_k_accuracy_score(y_true, y_score, k=5, labels=labels) if as_score else None,
        None, None,
        f1_score(y_true, y_pred, average='macro', zero_division=0),
        precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall_score(y_true, y_pred, average='macro', zero_division=0),
        f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    ]

    return performance


def mapping_elements(x:np.array, col_len:int) -> np.array:
    """Map elements as an array of col_len dimension.

    Args:
        x (np.array): elements to be mapped.
        col_len (int): shape of array.

    Returns:
        np.array: mapped elements.
    """
    x_array = np.zeros(col_len)
    vals, counts = np.unique(x, return_counts=True)
    np.place(x_array, np.isin(np.arange(col_len), vals), counts)
    return x_array


def update_scores(scores:np.ndarray, ids:list, duplicated:bool=True, func_name:str='mean') -> np.ndarray:
    """Update the scores, specially useful when a sample is represented in multiple subsamples 
        (i.e. with overflow).

    Args:
        scores (np.ndarray): scores to be updated.
        ids (list): sample ids.
        duplicated (bool, optional): duplicated ids (sentence divided into multiple chunks). 
            Defaults to True.
        func_name (str, optional): function to use to aggregate the scores. Defaults to 'mean'.

    Returns:
        np.ndarray: updated scores.
    """
    if duplicated:
        ids = np.array(ids)[range(0, len(ids), scores.shape[1])].reshape(-1, 1)
    else:
        ids = np.array(ids).reshape(-1, 1)
    if ids[-1, -1] == (scores.shape[0] - 1):
        return scores

    if func_name == 'max':
        scores_f = np.concatenate([ids, np.argmax(scores, axis=-1, keepdims=True)], axis=1)
        scores_f = np.split(scores_f[:,1], np.unique(scores_f[:,0], return_index=True)[1][1:])
        return np.vstack(list(map(lambda x: mapping_elements(x, scores.shape[1]), scores_f)))
    else:
        scores_f = np.concatenate([ids, scores], axis=1)
        scores_f = np.split(scores_f[:,1:], np.unique(scores_f[:,0], return_index=True)[1][1:])
        if func_name == 'median':
            return np.vstack(list(map(lambda x: np.median(x, axis=0), scores_f)))
        else:
            return np.vstack(list(map(lambda x: np.mean(x, axis=0), scores_f)))
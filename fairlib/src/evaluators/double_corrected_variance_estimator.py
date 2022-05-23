import pandas as pd
import numpy as np
from .evaluator import confusion_matrix_based_scores

def group_level_metrics(confusion_matrices, metric, class_id=1):
    """organizing the evaluation scores for each group.

    Args:
        confusion_matrices (dict): a dictionary of confusion_matrices of each group.
        metric (str): metic name, such as TPR and FPR.
        class_id (str): the class id of the metrics. Default to 1.

    Returns:
        pd.DataFrame: evaluation scores for each group with group size.
    """
    metric_df = []

    group_keys = list(confusion_matrices.keys())
    group_keys.remove("overall")

    for gid in group_keys:
        cnf_k = confusion_matrices[gid]
        if metric in ["TPR", "TNR"]:
            n_k = np.sum(cnf_k, axis=1)[class_id]
        elif metric in ["FPR", "FNR"]:
            n_k = np.sum(cnf_k) - np.sum(cnf_k, axis=1)[class_id]
        metric_k = confusion_matrix_based_scores(cnf_k)[metric][class_id]
        metric_df.append({
            "gid":gid, 
            "metric_k":metric_k, 
            "n_k":n_k,
        })
    metric_df = pd.DataFrame(metric_df)

    return metric_df

def double_correction(metric_df, n_sample = 1000, threshold = False, sample_variance = True):
    """Calculated corrected variance estimation.

    Args:
        metric_df (pd.DataFrame): results of group_level_metrics.
        n_sample (int, optional): number of trails in bootstrapping. Defaults to 1000.
        threshold (bool, optional): whether or not replace negative corrected variance to 0. Defaults to False.
        sample_variance (bool, optional): Use sample variance if true, and population variance otherwise. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing uncorrected_var, corrected_var, and double_corrected_var of all bootstrapping samples.
    """
    bootstrapping = []
    for _value in metric_df.values:
        bootstrapping.append(
            np.random.binomial(n=_value[2], p=_value[1], size=n_sample)/_value[2]
            )
    mu_hats = np.stack(bootstrapping)
    n_groups = len(metric_df.values)
    nks = metric_df.values[:,2]


    if sample_variance:
        uncorrected_var = np.var(mu_hats, axis=0)*n_groups/(n_groups-1)
    else:
        uncorrected_var = np.var(mu_hats, axis=0)
    sigma2_hats = (mu_hats*(1-mu_hats)/nks.reshape(-1,1))
    sigma2_hats_mean = np.mean(sigma2_hats, axis=0)
    corrected_var = uncorrected_var - sigma2_hats_mean
    double_corrected_var = uncorrected_var - 2*sigma2_hats_mean + np.mean(sigma2_hats/nks.reshape(-1,1), axis=0)

    if threshold:
        corrected_var = np.maximum(0, corrected_var)
        double_corrected_var = np.maximum(0, double_corrected_var)

    results_df = pd.DataFrame({
        "uncorrected_var":uncorrected_var,
        "corrected_var":corrected_var,
        "double_corrected_var":double_corrected_var,
    })

    return results_df

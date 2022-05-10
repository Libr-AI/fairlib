import os
from pathlib import Path
import torch
import yaml
from yaml.loader import SafeLoader
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from tqdm import tqdm 
import math
import seaborn as sns
import matplotlib.pyplot as plt


def mkdirs(paths):
    """make a set of new directories

    Args:
        paths (list): a list of directories
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """make a new directory

    Args:
        path (str): path to the directory
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def power_mean(series, p, axis=0):
    """calculate the generalized mean

    Args:
        series (np.array): a array of numbers.
        p (int): power of the generalized mean.
        axis (int, optional): axis to the aggregation. Defaults to 0.

    Returns:
        np.array: generalized mean of the inputs
    """
    if p>50:
        return np.max(series, axis=axis)
    elif p<-50:
        return np.min(series, axis=axis)
    else:
        total = np.mean(np.power(series, p), axis=axis)
        return np.power(total, 1 / p)

def l2norm(matrix_1, matrix_2):
    """calculate Euclidean distance

    Args:
        matrix_1 (n*d np array): n is the number of instances, d is num of metric
        matrix_2 (n*d np array): same as matrix_1

    Returns:
        float: the row-wise Euclidean distance 
    """
    return np.power(np.sum(np.power(matrix_1-matrix_2, 2), axis=1), 0.5)

def DTO(fairness_metric, performacne_metric, utopia_fairness = None, utopia_performance = None):
    """calculate DTO for each condidate model

    Args:
        fairness_metric (List): fairness evaluation results (1-GAP)
        performacne_metric (List): performance evaluation results
    """
    
    fairness_metric, performacne_metric = np.array(fairness_metric), np.array(performacne_metric)
    # Best metric
    if (utopia_performance is None):
        utopia_performance = np.max(performacne_metric)
    if (utopia_fairness is None):
        utopia_fairness = np.max(fairness_metric)

    # Normalize
    performacne_metric = performacne_metric/utopia_performance
    fairness_metric = fairness_metric/utopia_fairness

    # Reshape and concatnate
    performacne_metric = performacne_metric.reshape(-1,1)
    fairness_metric = fairness_metric.reshape(-1,1)
    normalized_metric = np.concatenate([performacne_metric, fairness_metric], axis=1)

    # Calculate Euclidean distance
    return l2norm(normalized_metric, np.ones_like(normalized_metric))

# find folders for each run
def get_dir(results_dir, project_dir, checkpoint_dir, checkpoint_name, model_id):
    """retrive logs for experiments

    Args:
        results_dir (str): dir to the saved experimental results
        project_dir (str): experiment type identifier, e.g., final, hypertune, dev. Same as the arguments.
        checkpoint_dir (str): dir to checkpoints, `models` by default.
        checkpoint_name (str):  checkpoint_epoch{num_epoch}.ptr.gz
        model_id (str): read all experiment start with the same model_id. E.g., "Adv" when tuning hyperparameters for standard adversarial

    Returns:
        list: a list of dictionaries, where each dict contains the information for a experiment.
    """
    results_dir = Path(results_dir)
    project_dir = Path(project_dir)
    checkpoint_dir = Path(checkpoint_dir)

    exps = []
    for root, dirs, files in os.walk(results_dir / project_dir):
        # for file in files:
            # if file.endswith(".txt"):
                # print(os.path.join(root, file))
        for dir in dirs:
            if dir.startswith(model_id):
                _dirs = Path(os.path.join(root, dir))

                # Open the file and load the file
                with open(_dirs / 'opt.yaml') as f:
                    _opt = yaml.load(f, Loader=SafeLoader)
                
                _checkpoints_dirs = []
                for root2, dirs2, files2 in os.walk(_dirs / checkpoint_dir):
                    for file2 in files2:
                        if file2.startswith(checkpoint_name):
                            _dirs2 = os.path.join(root2, file2)
                            _checkpoints_dirs.append(_dirs2)
                exps.append(
                    {
                        "opt":_opt,
                        "opt_dir":str(os.path.join(root, dir, 'opt.yaml')),
                        "dir":_checkpoints_dirs,
                    }
                )
    return exps

def get_model_scores(exp, GAP_metric, Performance_metric, keep_original_metrics = False):
    """given the log path for a exp, read log and return the dev&test performacne, fairness, and DTO

    Args:
        exp (str): get_dir output, includeing the options and path to checkpoints
        GAP_metric (str): the target GAP metric name
        Performance_metric (str): the target performance metric name, e.g., F1, Acc.

    Returns:
        pd.DataFrame: a pandas df including dev and test scores for each epoch
    """

    epoch_id = []
    epoch_scores_dev = {"performance":[],"fairness":[]}
    epoch_scores_test = {"performance":[],"fairness":[]}
    for epoch_result_dir in exp["dir"]:
        epoch_result = torch.load(epoch_result_dir)

        # Track the epoch id
        epoch_id.append(epoch_result["epoch"])

        # Get fairness evaluation scores, 1-GAP, the larger the better
        epoch_scores_dev["fairness"].append(1-epoch_result["dev_evaluations"][GAP_metric])
        epoch_scores_test["fairness"].append(1-epoch_result["test_evaluations"][GAP_metric])

        epoch_scores_dev["performance"].append(epoch_result["dev_evaluations"][Performance_metric])
        epoch_scores_test["performance"].append(epoch_result["test_evaluations"][Performance_metric])

        if keep_original_metrics:
            for _dev_keys in epoch_result["dev_evaluations"].keys():
                epoch_scores_dev[_dev_keys] = (epoch_scores_dev.get(_dev_keys,[]) + [epoch_result["dev_evaluations"][_dev_keys]])
            
            for _test_keys in epoch_result["test_evaluations"].keys():
                epoch_scores_test[_test_keys] = (epoch_scores_test.get(_test_keys,[]) + [epoch_result["test_evaluations"][_test_keys]])

    # Calculate the DTO for dev and test 
    dev_DTO = DTO(fairness_metric=epoch_scores_dev["fairness"], performacne_metric=epoch_scores_dev["performance"])
    test_DTO = DTO(fairness_metric=epoch_scores_test["fairness"], performacne_metric=epoch_scores_test["performance"])
    
    epoch_results_dict = {
            "epoch":epoch_id,
            "dev_DTO":dev_DTO,
            # "test_fairness":epoch_scores_test["fairness"],
            # "test_performance":epoch_scores_test["performance"],
            "test_DTO":test_DTO,
        }

    for _dev_metric_keys in epoch_scores_dev.keys():
        epoch_results_dict["dev_{}".format(_dev_metric_keys)] = epoch_scores_dev[_dev_metric_keys]
    for _test_metric_keys in epoch_scores_test.keys():
        epoch_results_dict["test_{}".format(_test_metric_keys)] = epoch_scores_test[_test_metric_keys]

    epoch_scores = pd.DataFrame(epoch_results_dict)

    return epoch_scores

def retrive_all_exp_results(exp,GAP_metric_name, Performance_metric_name,index_column_names, keep_original_metrics = False):
    """retrive experimental results according to the input list of experiments.

    Args:
        exp (list): a list of experiment info.
        GAP_metric_name (str): gap metric name, such as TPR_GAP.
        Performance_metric_name (str): performance metric name, such as accuracy.
        index_column_names (_type_): a list of hyperparameters that will be used to differentiate runs within a single method.
        keep_original_metrics (bool): whether or not keep all experimental results in the checkpoint. Default to False. 

    Returns:
        dict: retrived results.
    """
    # Get scores
    epoch_scores = get_model_scores(
        exp=exp, GAP_metric=GAP_metric_name, 
        Performance_metric=Performance_metric_name,
        keep_original_metrics = keep_original_metrics,
        )
    _exp_results = epoch_scores

    _exp_opt = exp["opt"]
    # Get hyperparameters for this epoch
    for hyperparam_key in index_column_names:
        _exp_results[hyperparam_key] = [_exp_opt[hyperparam_key]]*len(_exp_results)
    
    _exp_results["opt_dir"] = [exp["opt_dir"]]*len(_exp_results)

    return _exp_results

def retrive_exp_results(
    exp,GAP_metric_name, Performance_metric_name,
    selection_criterion,index_column_names, keep_original_metrics = False):
    """Retrive experimental results of a epoch from the saved checkpoint.

    Args:
        exp (_type_): _description_
        GAP_metric_name (_type_): _description_
        Performance_metric_name (_type_): _description_
        selection_criterion (_type_): _description_
        index_column_names (_type_): _description_
        keep_original_metrics (bool, optional): besides selected performance and fairness, show original metrics. Defaults to False.

    Returns:
        dict: retrived results.
    """

    # Get scores
    epoch_scores = get_model_scores(
        exp=exp, GAP_metric=GAP_metric_name, 
        Performance_metric=Performance_metric_name,
        keep_original_metrics=keep_original_metrics,
        )
    if selection_criterion == "DTO":
        selected_epoch_id = np.argmin(epoch_scores["dev_{}".format(selection_criterion)])
    else:
        selected_epoch_id = np.argmax(epoch_scores["dev_{}".format(selection_criterion)])
    selected_epoch_scores = epoch_scores.iloc[selected_epoch_id]

    _exp_opt = exp["opt"]

    # Get hyperparameters for this epoch
    _exp_results = {}
    for hyperparam_key in index_column_names:
        _exp_results[hyperparam_key] = _exp_opt[hyperparam_key]

    # Merge opt with scores
    for key in selected_epoch_scores.keys():
        _exp_results[key] = selected_epoch_scores[key]
    
    _exp_results["opt_dir"] = exp["opt_dir"]
    _exp_results["epoch"] = selected_epoch_id

    return _exp_results

def is_pareto_efficient(costs, return_mask = True):
    """Find the pareto-efficient points

    If return_mask is True, this will be an (n_points, ) boolean array
    Otherwise it will be a (n_efficient_points, ) integer array of indices.

    Args:
        costs (np.array): An (n_points, n_costs) array
        return_mask (bool, optional): True to return a mask. Defaults to True.

    Returns:
        np.array: An array of indices of pareto-efficient points.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def auc_performance_fairness_tradeoff(
    pareto_df,
    random_performance = 0, 
    pareto_selection = "test",
    fairness_metric_name = "fairness",
    performance_metric_name = "performance",
    interpolation = "linear",
    performance_threshold = None,
    normalization = False,
    ):
    """calculate the area under the performance--fairness trade-off curve.

    Args:
        pareto_df (DataFrame): A data frame of pareto frontiers
        random_performance (float, optional): the lowest performance, which leads to the 1 fairness. Defaults to 0.
        pareto_selection (str, optional): which split is used to select the frontiers. Defaults to "test".
        fairness_metric_name (str, optional):. the metric name for fairness evaluation. Defaults to "fairness".
        performance_metric_name (str, optional): the metric name for performance evaluation. Defaults to "performance".
        interpolation (str, optional): interpolation method for the threshold fairness. Defaults to "linear".
        performance_threshold (float, optional): the performance threshold for the method. Defaults to None.
        normalization (bool, optional): if normalize the auc score with the maximum auc that can be achieved. Defaults to False.

    Returns:
        tuple: (AUC score, AUC DataFrame)
    """
    fairness_col_name = "{}_{} mean".format(pareto_selection, fairness_metric_name)
    performance_col_name = "{}_{} mean".format(pareto_selection, performance_metric_name)

    # Filter the df with only performance and fairness scores
    results_df = pareto_df[[fairness_col_name, performance_col_name]]

    # Add the worst performed model
    results_df = results_df.append({
        fairness_col_name: 1,
        performance_col_name: random_performance,
        }, ignore_index=True)

    sorted_results_df = results_df.sort_values(by=[fairness_col_name])

    if performance_threshold is not None:
        if performance_threshold > sorted_results_df.values[0][1]:
            return 0, None
        if performance_threshold < sorted_results_df.values[-1][1]:
            performance_threshold = sorted_results_df.values[-1][1]

        # Find the closest performed points to the threshold
        closest_worser_performed_point =  sorted_results_df[sorted_results_df[performance_col_name]<=performance_threshold].values[0]
        closest_better_performed_point =  sorted_results_df[sorted_results_df[performance_col_name]>=performance_threshold].values[-1]

        # Interpolation
        assert interpolation in ["linear", "constant"]
        if interpolation == "constant":
            if (performance_threshold-closest_worser_performed_point[1]) <= (closest_better_performed_point[1]-performance_threshold):
                interpolation_fairness = closest_worser_performed_point[0]
            else:
                interpolation_fairness = closest_better_performed_point[0]
        elif interpolation == "linear":
            _ya, _xa = closest_worser_performed_point[0], closest_worser_performed_point[1]
            _yb, _xb = closest_better_performed_point[0], closest_better_performed_point[1]
            interpolation_fairness = _ya+(_yb-_ya)*((performance_threshold-_xa)/(_xb-_xa))

        interpolated_point = {
                fairness_col_name: interpolation_fairness,
                performance_col_name: performance_threshold,
            }
        
        sorted_results_df = sorted_results_df[sorted_results_df[performance_col_name]>=performance_threshold]
        sorted_results_df = sorted_results_df.append(
            interpolated_point, ignore_index=True,
        )


    filtered_curve = sorted_results_df.sort_values(by=[performance_col_name])
    auc_filtered_curve = np.trapz(
        filtered_curve[fairness_col_name], 
        x=filtered_curve[performance_col_name], )

    # Calculate the normalization term
    if normalization:
        normalization_term = (1-min(filtered_curve[performance_col_name]))
        auc_filtered_curve = auc_filtered_curve / normalization_term

    return auc_filtered_curve, filtered_curve
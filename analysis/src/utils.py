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


def power_mean(series, p):
    if p>50:
        return max(series)
    elif p<50:
        return min(series)
    else:
        total = np.mean(np.power(series, p))
        return np.power(total, 1 / p)

def l2norm(matrix_1, matrix_2):
    """calculate Euclidean distance

    Args:
        matrix_1 (n*d np array): n is the number of instances, d is num of metric
        matrix_2 (n*d np array): same as matrix_1

    Returns:
        _type_: _description_
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
        model_id (_type_): read all experiment start with the same model_id. E.g., "Adv" when tuning hyperparameters for standard adversarial

    Returns:
        _type_: _description_
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
                        "dir":_checkpoints_dirs,
                    }
                )
    return exps

def get_model_scores(exp, GAP_metric, Performance_metric):
    """given the log path for a exp, read log and return the dev&test performacne, fairness, and DTO

    Args:
        exp (_type_): get_dir output, includeing the options and path to checkpoints
        GAP_metric (_type_): the target GAP metric name
        Performance_metric (_type_): the target performance metric name, e.g., F1, Acc.

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

        # Get evaluation scores
        epoch_scores_dev["fairness"].append(1-epoch_result["dev_evaluations"][GAP_metric])
        epoch_scores_test["fairness"].append(1-epoch_result["test_evaluations"][GAP_metric])

        epoch_scores_dev["performance"].append(epoch_result["dev_evaluations"][Performance_metric])
        epoch_scores_test["performance"].append(epoch_result["test_evaluations"][Performance_metric])

        # Calculate the DTO for dev and test 
    dev_DTO = DTO(fairness_metric=epoch_scores_dev["fairness"], performacne_metric=epoch_scores_dev["performance"])
    test_DTO = DTO(fairness_metric=epoch_scores_test["fairness"], performacne_metric=epoch_scores_test["performance"])

    epoch_scores = pd.DataFrame(
        {
            "epoch":epoch_id,
            "dev_{}".format(GAP_metric):epoch_scores_dev["fairness"],
            "dev_{}".format(Performance_metric):epoch_scores_dev["performance"],
            "dev_DTO":dev_DTO,
            "test_{}".format(GAP_metric):epoch_scores_test["fairness"],
            "test_{}".format(Performance_metric):epoch_scores_test["performance"],
            "test_DTO":test_DTO,
        }
    )

    return epoch_scores


def model_selection(
    results_dir,
    project_dir,
    checkpoint_dir,
    checkpoint_name,
    model_id,
    GAP_metric_name,
    Performance_metric_name,
    selection_criterion,
    index_column_names = ['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda']
    ):
    """perform model selection over different runs wrt different hyperparameters

    Args:
        results_dir (str): dir to the saved experimental results
        project_dir (str): experiment type identifier, e.g., final, hypertune, dev. Same as the arguments.
        checkpoint_dir (str): dir to checkpoints, `models` by default.
        checkpoint_name (str):  checkpoint_epoch{num_epoch}.ptr.gz
        model_id (str): read all experiment start with the same model_id. E.g., "Adv" when tuning hyperparameters for standard adversarial
        GAP_metric_name (str): fairness metric in the log
        Performance_metric_name (str): performance metric name in the log
        selection_criterion (str): {GAP_metric_name | Performance_metric_name | "DTO"}
        index_column_names (list): tuned hyperparameters, ['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda'] by default.

    Returns:
        _type_: loaded results as a dataframe for different runs
    """
    exps = get_dir(
        results_dir=results_dir, 
        project_dir=project_dir, 
        checkpoint_dir=checkpoint_dir, 
        checkpoint_name=checkpoint_name, 
        model_id=model_id)

    exp_results = []
    for exp in tqdm(exps):
        # Get scores
        epoch_scores = get_model_scores(exp=exp, GAP_metric=GAP_metric_name, Performance_metric=Performance_metric_name)
        selected_epoch_id = np.argmin(epoch_scores["dev_{}".format(selection_criterion)])
        selected_epoch_scores = epoch_scores.iloc[selected_epoch_id]

        _exp_opt = exp["opt"]

        # Get hyperparameters for this epoch
        _exp_results = {}
        for hyperparam_key in index_column_names:
            _exp_results[hyperparam_key] = _exp_opt[hyperparam_key]

        # Merge opt with scores
        for key in selected_epoch_scores.keys():
            _exp_results[key] = selected_epoch_scores[key]

        exp_results.append(_exp_results)
    return pd.DataFrame(exp_results)

def retrive_exp_results(exp,GAP_metric_name, Performance_metric_name,selection_criterion,index_column_names):
    # Get scores
    epoch_scores = get_model_scores(exp=exp, GAP_metric=GAP_metric_name, Performance_metric=Performance_metric_name)
    selected_epoch_id = np.argmin(epoch_scores["dev_{}".format(selection_criterion)])
    selected_epoch_scores = epoch_scores.iloc[selected_epoch_id]

    _exp_opt = exp["opt"]

    # Get hyperparameters for this epoch
    _exp_results = {}
    for hyperparam_key in index_column_names:
        _exp_results[hyperparam_key] = _exp_opt[hyperparam_key]

    # Merge opt with scores
    for key in selected_epoch_scores.keys():
        _exp_results[key] = selected_epoch_scores[key]

    return _exp_results

def create_plots(
    input_df, 
    key_index = 0, 
    index_column_names = ['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda'],
    GAP_metric_name = "rms_TPR",
    Performance_metric_name = "accuracy",
    ):
    """create plots and return the selected model

    Args:
        input_df (pd.DataFrame): results dataframe

    Returns:
        _type_: selected model
    """
    # Moji_adv_df
    _df = input_df.set_index(index_column_names)

    __df = _df.groupby(_df.index).agg(["mean", "var"]).reset_index()
    __df

    _log_lambda = [round(math.log10(i[key_index]), 2) for i in __df["index"]]
    __df["log_lambda"] = _log_lambda

    _final_DTO = DTO(
        fairness_metric=list(__df[("test_{}".format(GAP_metric_name), "mean")]), 
        performacne_metric=list(__df[("test_{}".format(Performance_metric_name), "mean")])
        )
    __df["final_DTO"] = _final_DTO

    sns.relplot(data=__df, x="log_lambda", y="final_DTO")

    sns.relplot(
        data=__df, 
        x=("test_{}".format(Performance_metric_name), "mean"), 
        y=("test_{}".format(GAP_metric_name), "mean"))

    return __df[__df["final_DTO"] == min(_final_DTO)]
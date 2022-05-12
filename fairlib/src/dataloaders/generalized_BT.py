import itertools
import numpy as np
from random import choices

def get_data_distribution(y_data, g_data):
    """Given target label and protected labels, calculate empirical distributions.

    joint_dist: n_class * n_groups matrix, where each element refers to the joint probability, i.e., proportion size.
    g_dist: n_groups array, indicating the prob of each group
    y_dist: n_class array, indicating the prob of each class
    g_cond_y_dit: n_class * n_groups matrix, g_cond_y_dit[y_id,:] refers to the group distribution within class y_id
    y_cond_g_dit: n_class * n_groups matrix, y_cond_g_dit[:,g_id] refers to the class distribution within group g_id

    Args:
        y_data (np.ndarray): target labels
        g_data (np.ndarray): protected labels

    Returns:
        dict: a dict of distribution info.
    """
    assert isinstance(y_data,np.ndarray), "the data type is expected to be array"
    assert isinstance(g_data,np.ndarray), "the data type is expected to be array"

    # Get distinct items of target labels and group labels
    g_item = list(set(g_data))
    y_item = list(set(y_data))
    yg_tuple = list(itertools.product(y_item, g_item))

    # Masks
    yg_mask = {}

    for (tmp_y, tmp_g) in yg_tuple:
        yg_mask[(tmp_y, tmp_g)] = ((y_data == tmp_y) & (g_data == tmp_g))
    
    # Indices
    yg_index = {}
    for tmp_yg in yg_tuple:
            yg_index[tmp_yg] = np.flatnonzero(yg_mask[tmp_yg])
    
    # n_class * n_groups saving the empirical probs of each partition
    count_matrix = np.zeros((len(y_item), len(g_item)))
    
    for tmp_yg in yg_tuple:
        count_matrix[tmp_yg] = len(yg_index[tmp_yg])
    
    # normalize the count to prob, i.e., sum to 1
    N_instacnes = np.sum(count_matrix)
    joint_dist = count_matrix / N_instacnes

    g_dist = np.sum(joint_dist,axis=0)
    y_dist = np.sum(joint_dist,axis=1)

    g_cond_y_dist = joint_dist / y_dist.reshape(-1,1)
    y_cond_g_dist = joint_dist / g_dist.reshape(1,-1)
    
    distribution_dict = {
        "joint_dist":joint_dist, 
        "g_dist":g_dist, 
        "y_dist":y_dist,
        "g_cond_y_dist":g_cond_y_dist,
        "y_cond_g_dist":y_cond_g_dist,
        "yg_index":yg_index,
        "N":N_instacnes,
    }

    return distribution_dict

def generalized_sampling(
    default_distribution_dict,
    N=None, joint_dist=None, g_dist=None, y_dist=None, 
    g_cond_y_dist=None,y_cond_g_dist=None):
    """Perform resampling according to the specified distribution information

    Args:
        default_distribution_dict (dict): a dict of distribution information of the original dataset.
        N (int, optional): The total number of returned indices. Defaults to None.
        joint_dist (np.ndarray, optional): n_class * n_groups matrix, where each element refers to the joint probability, i.e., proportion size. Defaults to None.
        g_dist (np.ndarray, optional): n_groups array, indicating the prob of each group. Defaults to None.
        y_dist (np.ndarray, optional): n_class array, indicating the prob of each class. Defaults to None.
        g_cond_y_dist (np.ndarray, optional): n_class * n_groups matrix, g_cond_y_dit[y_id,:] refers to the group distribution within class y_id. Defaults to None.
        y_cond_g_dist (np.ndarray, optional): n_class * n_groups matrix, y_cond_g_dit[:,g_id] refers to the class distribution within group g_id. Defaults to None.

    Returns:
        list: list of selected indices.
    """

    if joint_dist is not None:
        assert joint_dist.shape == default_distribution_dict["joint_dist"].shape
        target_joint_dist = joint_dist
    else:
        target_joint_dist = default_distribution_dict["joint_dist"]

    if g_dist is not None:
        assert g_dist.shape == default_distribution_dict["g_dist"].shape
        target_g_dist = g_dist
    else:
        target_g_dist = default_distribution_dict["g_dist"]

    if y_dist is not None:
        assert y_dist.shape == default_distribution_dict["y_dist"].shape
        target_y_dist = y_dist
    else:
        target_y_dist = default_distribution_dict["y_dist"]

    if g_cond_y_dist is not None:
        assert g_cond_y_dist.shape == default_distribution_dict["g_cond_y_dist"].shape
        target_g_cond_y_dist = g_cond_y_dist
    else:
        target_g_cond_y_dist = default_distribution_dict["g_cond_y_dist"]

    if y_cond_g_dist is not None:
        assert y_cond_g_dist.shape == default_distribution_dict["y_cond_g_dist"].shape
        target_y_cond_g_dist = y_cond_g_dist
    else:
        target_y_cond_g_dist = default_distribution_dict["y_cond_g_dist"]

    # 1st priority: joint_dict
    if joint_dist is not None:
        target_dist = target_joint_dist
    
    # 2nd priority: y_dist and g_cond_y_dist
    elif (y_dist is not None) or (g_cond_y_dist is not None):
        target_dist = target_g_cond_y_dist * target_y_dist.reshape(-1,1)
    
    # 3rd priority: g_dist and y_cond_g_dist
    elif (g_dist is not None) or (y_cond_g_dist is not None):
        target_dist = target_y_cond_g_dist * target_g_dist.reshape(1,-1)

    # If all arguments are None, use the original distribution for sampling
    else:
        target_dist = default_distribution_dict["joint_dist"]
    
    assert np.min(target_dist) >= 0, "probs mush be non-negative, check the manipulation function"
    # Make sure that the sum of probs is 1
    target_dist = target_dist / np.sum(target_dist)

    # Calculate the number of instacnes for each subset
    if N is not None:
        target_dist = target_dist * int(N)
    else:
        target_dist = target_dist * default_distribution_dict["N"]
    
    selected_index = []
    # Sampling
    for _yg_tuple, _yg_index in default_distribution_dict["yg_index"].items():
        _target_N = int(round(target_dist[_yg_tuple]))

        if _target_N <= len(_yg_index):
            selected_index = selected_index + list(_yg_index[:_target_N])
        else:
            selected_index = selected_index + list(_yg_index)
            selected_index = selected_index + choices(_yg_index, k=(_target_N - len(_yg_index)))
    
    return selected_index

def manipulate_data_distribution(default_distribution_dict, N = None, GBTObj = "original", alpha = 1):
    """generalized BT

    Args:
        default_distribution_dict (dict): a dict of distribution information of the original dataset.
        N (int, optional): The total number of returned indices. Defaults to None.
        GBTObj (str, optional): original | joint | g | y | g_cond_y | y_cond_g. Defaults to "original".
        alpha (int, optional): interpolation between the original distribution and the target distribution. Defaults to 1.

    Returns:
        list: list of selected indices.
    """

    if GBTObj == "original":
        return generalized_sampling(default_distribution_dict, N)
    elif GBTObj == "joint":
        target_joint_dist = np.ones_like(default_distribution_dict["joint_dist"])
        target_joint_dist = target_joint_dist / (target_joint_dist.shape[0]*target_joint_dist.shape[1])
        target_joint_dist = target_joint_dist * alpha + (1-alpha) * default_distribution_dict["joint_dist"]
        return generalized_sampling(default_distribution_dict, N, joint_dist = target_joint_dist)
    elif GBTObj == "g":
        target_g_dist = np.ones_like(default_distribution_dict["g_dist"])
        target_g_dist = target_g_dist / len(target_g_dist)
        target_g_dist = target_g_dist * alpha + (1-alpha) * default_distribution_dict["g_dist"]
        return generalized_sampling(default_distribution_dict, N, g_dist = target_g_dist)
    elif GBTObj == "y":
        target_y_dist = np.ones_like(default_distribution_dict["y_dist"])
        target_y_dist = target_y_dist / len(target_y_dist)
        target_y_dist = target_y_dist * alpha + (1-alpha) * default_distribution_dict["y_dist"]
        print(target_y_dist)
        return generalized_sampling(default_distribution_dict, N, y_dist = target_y_dist)
    elif GBTObj == "g_cond_y":
        target_g_cond_y_dist = np.ones_like(default_distribution_dict["g_cond_y_dist"])
        target_g_cond_y_dist = target_g_cond_y_dist / (target_g_cond_y_dist.shape[1]*np.ones((1,target_g_cond_y_dist.shape[1])))
        target_g_cond_y_dist = target_g_cond_y_dist * alpha + (1-alpha) * default_distribution_dict["g_cond_y_dist"]
        print(target_g_cond_y_dist)
        return generalized_sampling(default_distribution_dict, N, g_cond_y_dist = target_g_cond_y_dist)
    elif GBTObj == "y_cond_g":
        target_y_cond_g_dist = np.ones_like(default_distribution_dict["y_cond_g_dist"])
        target_y_cond_g_dist = target_y_cond_g_dist / (target_y_cond_g_dist.shape[0]*np.ones((target_y_cond_g_dist.shape[0],1)))
        target_y_cond_g_dist = target_y_cond_g_dist * alpha + (1-alpha) * default_distribution_dict["y_cond_g_dist"]
        return generalized_sampling(default_distribution_dict, N, y_cond_g_dist = target_y_cond_g_dist)
    else:
        raise NotImplementedError
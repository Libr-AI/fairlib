import itertools
import numpy as np

def get_data_distribution(y_data, g_data):
    """Given target label and protected labels, calculate emprical distributions.

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
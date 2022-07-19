from itertools import combinations
import numpy as np

def task_comb_data(df, 
                task_combs, 
                conditions,):
    """Return selected rows that meets the requirements

    Args:
        df (pd.DataFrame): data frame
        task_combs (list): column names
        conditions (list): one value for each column

    Returns:
        list: indicators of selection
    """
    selected_rows = np.array([True]*len(df))
    for task, condition in zip(task_combs, conditions):
        selected_rows = selected_rows & (df[task].to_numpy()==condition)
    return selected_rows

def get_all_combs(unique_type_list):
    number_tasks = len(unique_type_list)
    no_unique_types = [len(unique_type) for unique_type in unique_type_list]+[1]
    total_number = np.prod(no_unique_types)
    # print(number_tasks, total_number)
    
    # init 2d matrix
    group_combinations = [[None for j in range(number_tasks)] for i in range(total_number)]

    for single_task_id, single_task_types in enumerate(unique_type_list):
        # print(single_task_id, single_task_types)

        # calculate single repeat time
        single_repeat_time = int(np.prod(no_unique_types[single_task_id+1:]))
        # calculate whole list repeat time
        whole_repeat_time = int(total_number/single_repeat_time/len(single_task_types))
        # print(single_repeat_time, whole_repeat_time)

        # create col number
        task_col = []
        # single repeat
        for t_type in single_task_types:
            task_col = task_col + [t_type]*single_repeat_time
        # whole repeat
        task_col = task_col * whole_repeat_time
        # print(task_col)

        # add to matrix
        for i, v in enumerate(task_col):
            group_combinations[i][single_task_id] = v
    return group_combinations

# Gerry Fairness
def Gerrymandering_groups(attributes, attribute_distinct_labels):
    '''
    attributes: a list of attribute names, e.g., "gender", "age"
    attribute_distinct_labels: a dictionary where each value is the distinct label list corresponding to the key.
    '''
    attribute_label_pairs = []
    # iterate all combinations of attributes
    for l in range(len(attributes)):
        for attribute_comb in [list(i) for i in combinations(attributes, (l+1))]:
            comb_distinct_labels = [attribute_distinct_labels[comb] for comb in attribute_comb]
            all_att_comb_label_combs = get_all_combs(comb_distinct_labels)
            for att_label in  all_att_comb_label_combs:
                attribute_label_pairs.append((attribute_comb, att_label))

    return attribute_label_pairs

if __name__ == '__main__':
    import pandas as pd

    data_df = pd.DataFrame({
            "age" : [["Old", "Young"][i] for i in np.random.randint(0,2,size=100)],
            "race" : [["Black", "White"][i] for i in np.random.randint(0,2,size=100)],
            "gender" : [["Female", "Male"][i] for i in np.random.randint(0,2,size=100)]
        })
    print(data_df)

    gerry_group = Gerrymandering_groups(
        attributes= ["age", "gender", "race"],
        attribute_distinct_labels = {
            "age" : ["Old", "Young"], 
            "race" : ["Black", "White"], 
            "gender" : ["Female", "Male"]
        }
    )


    group_id = 10
    print(gerry_group[group_id])
    print(data_df[task_comb_data(data_df, gerry_group[group_id][0], gerry_group[group_id][1])])
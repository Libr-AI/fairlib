# Manilulating Data Distribution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HanXudong/fairlib/blob/main/tutorial/manipulate_data_distribution.ipynb)

In this notebook, we will introduce the built-in function for manipulating data distributions.

Overall, the process can be described as:
1. Load data from files
2. Analysis the loaded dataset distribution
3. Resample isntacnes based on their target labels and protected labels

Implementations for steps 1 and 4 are the same with others. Please see [dataloader](https://hanxudong.github.io/fairlib/reference_api_dataloaders.html) for detailed description.

## Load data

Similar to long-tail leaning, the data distribution is essentially the label distributions. However, besides the target label in long-tail learning, we also need to consider the demographic labels.

Since the input distribution is not required for manipulating data distributions, we just create synthetic labels for demonstration purpose.


```python
import numpy as np

y = []
g = []
for _y, _g, _n in zip(
    [1,1,1,0,0,0], # Target labels, y
    [2,1,0,1,2,0], # Protected labels, g
    [4,5,6,3,7,9]  # Number of instances with target label _y and group label _g
    ):
    y = y + [_y]*_n
    g = g + [_g]*_n

y = np.array(y)
g = np.array(g)
```

## Analysis the loaded dataset distribution

Essentially, we are interested in the joint distribution of target label *y* and protected label *g*, which are both discrete random variables.

We save the corresponding results in probability tables:

Given target label and protected labels, calculate empirical distributions.

- joint_dist: n_class * n_groups matrix, where each element refers to the joint probability, i.e., proportion size.
- g_dist: n_groups array, indicating the prob of each group
- y_dist: n_class array, indicating the prob of each class
- g_cond_y_dist: n_class * n_groups matrix, g_cond_y_dit[y_id,:] refers to the group distribution within class y_id
- y_cond_g_dist: n_class * n_groups matrix, y_cond_g_dit[:,g_id] refers to the class distribution within group g_id


```python
from fairlib.src.dataloaders.generalized_BT import get_data_distribution
```


```python
synthetic_data_distribution = get_data_distribution(y_data = y, g_data = g)
print(synthetic_data_distribution.keys())
```

    dict_keys(['joint_dist', 'g_dist', 'y_dist', 'g_cond_y_dist', 'y_cond_g_dist', 'yg_index', 'N'])
    


```python
# n_class * n_groups matrix, where each element refers to the joint probability
synthetic_data_distribution["joint_dist"]
```




    array([[0.26470588, 0.08823529, 0.20588235],
           [0.17647059, 0.14705882, 0.11764706]])



## Resample isntacnes based on their target labels and protected labels

In order to specify a target distribution of *y* and *g*, there are 5 options in the `generalized_sampling` function, which can be classified into 3 types as:
1. specify the target joint distribution of *y* and *g*, i.e., *p(y,g)*.
2. specify the y_dist (*p(y)*) and g_cond_y_dist (*p(g|y)*), and *p(y,g) = p(g|y)p(y)*
3. specify the g_dist (*p(g)*) and y_cond_g_dist (*p(y|g)*), and *p(y,g) = p(y|g)p(g)*


```python
from fairlib.src.dataloaders.generalized_BT import generalized_sampling
```

Taking the group label distribution as an example, we would like the group label is uniformly distributed, i.e., *p(g=0)=p(g=1)=p(g=2)*


```python
target_g_dist = np.ones(3)/3
target_g_dist
```




    array([0.33333333, 0.33333333, 0.33333333])




```python
balanced_g_indices = generalized_sampling(
    default_distribution_dict = synthetic_data_distribution,
    N = 20,
    g_dist=target_g_dist,)
print(balanced_g_indices)
```

    [25, 26, 27, 28, 15, 16, 17, 18, 19, 20, 21, 9, 10, 11, 4, 5, 6, 7, 0, 1]
    


```python
balanced_g_distribution = get_data_distribution(
    y_data = y[balanced_g_indices], 
    g_data = g[balanced_g_indices])
```


```python
balanced_g_distribution
```




    {'joint_dist': array([[0.2 , 0.15, 0.2 ],
            [0.15, 0.2 , 0.1 ]]),
     'g_dist': array([0.35, 0.35, 0.3 ]),
     'y_dist': array([0.55, 0.45]),
     'g_cond_y_dist': array([[0.36363636, 0.27272727, 0.36363636],
            [0.33333333, 0.44444444, 0.22222222]]),
     'y_cond_g_dist': array([[0.57142857, 0.42857143, 0.66666667],
            [0.42857143, 0.57142857, 0.33333333]]),
     'yg_index': {(0, 0): array([0, 1, 2, 3], dtype=int64),
      (0, 1): array([4, 5, 6], dtype=int64),
      (0, 2): array([ 7,  8,  9, 10], dtype=int64),
      (1, 0): array([11, 12, 13], dtype=int64),
      (1, 1): array([14, 15, 16, 17], dtype=int64),
      (1, 2): array([18, 19], dtype=int64)},
     'N': 20.0}



Theoretically, we could sample any data distributions with the `generalized_sampling` function. But it is inconvenient to specify the distribution array every time. We further provide the `manipulate_data_distribution` function to simplify the manipulation process.

Specifically, the manipulation is based on the interpolation between the original data distribution and a specifically balanced target distribution. These balanced target distributions are identical to [**BT**](https://hanxudong.github.io/fairlib/reference_api_debiasing/BT.html).

The *alpha* refers to the interpolation extent as $final\_dist = \alpha*balanced\_dist + (1-\alpha)*origianl\_dist$

- default_distribution_dict (dict): a dict of distribution information of the original dataset.
- N (int, optional): The total number of returned indices. Defaults to None.
- GBTObj (str, optional): original | joint | g | y | g_cond_y | y_cond_g. Defaults to "original".
- alpha (int, optional): interpolation between the original distribution and the target distribution. Defaults to 1.


```python
from fairlib.src.dataloaders.generalized_BT import manipulate_data_distribution
```

The following examples show to what extent can we manipulate the distribution by tuning the alpha value
- aphla > 1: anto-imbalance.
- aphla = 1: perfectly balanced distribution as corresponding BT objectives.
- 0 < aphla < 1: interpolation between original distribution and the perfectly balanced distribution.
- alpha = 0: original distribution.
- alpha < 0: amplify imbalance.


```python
g_indices_10 = manipulate_data_distribution(default_distribution_dict = synthetic_data_distribution, N = 20, GBTObj = "g", alpha =1)
g_indices_00 = manipulate_data_distribution(default_distribution_dict = synthetic_data_distribution, N = 20, GBTObj = "g", alpha =0)
g_indices_05 = manipulate_data_distribution(default_distribution_dict = synthetic_data_distribution, N = 20, GBTObj = "g", alpha =0.5)
g_indices_n10 = manipulate_data_distribution(default_distribution_dict = synthetic_data_distribution, N = 20, GBTObj = "g", alpha =-1)
```


```python
print("Original:", synthetic_data_distribution["g_dist"])
print("Balanced g:", get_data_distribution(y_data = y[balanced_g_indices], g_data = g[balanced_g_indices])["g_dist"])
print("aplha 0.0:", get_data_distribution(y_data = y[g_indices_00], g_data = g[g_indices_00])["g_dist"])
print("aplha 0.5:", get_data_distribution(y_data = y[g_indices_05], g_data = g[g_indices_05])["g_dist"])
print("aplha 1.0:", get_data_distribution(y_data = y[g_indices_10], g_data = g[g_indices_10])["g_dist"])
print("aplha -1.0:", get_data_distribution(y_data = y[g_indices_n10], g_data = g[g_indices_n10])["g_dist"])
```

    Original: [0.44117647 0.23529412 0.32352941]
    Balanced g: [0.35 0.35 0.3 ]
    aplha 0.0: [0.45 0.25 0.3 ]
    aplha 0.5: [0.4 0.3 0.3]
    aplha 1.0: [0.35 0.35 0.3 ]
    aplha -1.0: [0.55 0.15 0.3 ]
    

## Limitation and Extension

The `manipulate_data_distribution` function strongly assumes that the original distribution is **NOT** identical to the target perfectly balanced distribution. For those with a perfectly balanced dataset, the target distributions will need to be manually specified and use the `generalized_sampling` for resampling.

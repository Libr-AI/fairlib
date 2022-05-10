# Customizing Evaluation Metrics


## Installation


```python
!pip install fairlib
```

    Collecting fairlib
      Downloading fairlib-0.0.3-py3-none-any.whl (63 kB)
    [K     |████████████████████████████████| 63 kB 2.0 MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.21.6)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.0.2)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.7/dist-packages (from fairlib) (0.11.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from fairlib) (4.64.0)
    Requirement already satisfied: docopt in /usr/local/lib/python3.7/dist-packages (from fairlib) (0.6.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.3.5)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from fairlib) (3.13)
    Collecting pickle5
      Downloading pickle5-0.0.12-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (256 kB)
    [K     |████████████████████████████████| 256 kB 29.1 MB/s 
    [?25hRequirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from fairlib) (3.2.2)
    Collecting transformers
      Downloading transformers-4.18.0-py3-none-any.whl (4.0 MB)
    [K     |████████████████████████████████| 4.0 MB 55.2 MB/s 
    [?25hRequirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.11.0+cu113)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (1.4.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (3.0.8)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->fairlib) (4.2.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib->fairlib) (1.15.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->fairlib) (2022.1)
    Requirement already satisfied: scipy>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (1.4.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (3.1.0)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (1.1.0)
    Collecting sacremoses
      Downloading sacremoses-0.0.53.tar.gz (880 kB)
    [K     |████████████████████████████████| 880 kB 51.9 MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (2.23.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (3.6.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (4.11.3)
    Collecting huggingface-hub<1.0,>=0.1.0
      Downloading huggingface_hub-0.5.1-py3-none-any.whl (77 kB)
    [K     |████████████████████████████████| 77 kB 6.9 MB/s 
    [?25hCollecting PyYAML
      Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
    [K     |████████████████████████████████| 596 kB 60.7 MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (2019.12.20)
    Collecting tokenizers!=0.11.3,<0.13,>=0.11.1
      Downloading tokenizers-0.12.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)
    [K     |████████████████████████████████| 6.6 MB 54.8 MB/s 
    [?25hRequirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (21.3)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers->fairlib) (3.8.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (2021.10.8)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (3.0.4)
    Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers->fairlib) (7.1.2)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.53-py3-none-any.whl size=895260 sha256=c5c48dec8f73ec0d60ebce7e2289c19af0e22809669612f2359f236a871759ad
      Stored in directory: /root/.cache/pip/wheels/87/39/dd/a83eeef36d0bf98e7a4d1933a4ad2d660295a40613079bafc9
    Successfully built sacremoses
    Installing collected packages: PyYAML, tokenizers, sacremoses, huggingface-hub, transformers, pickle5, fairlib
      Attempting uninstall: PyYAML
        Found existing installation: PyYAML 3.13
        Uninstalling PyYAML-3.13:
          Successfully uninstalled PyYAML-3.13
    Successfully installed PyYAML-6.0 fairlib-0.0.3 huggingface-hub-0.5.1 pickle5-0.0.12 sacremoses-0.0.53 tokenizers-0.12.1 transformers-4.18.0
    


```python
import fairlib
```


```python
!mkdir -p data/deepmoji
!wget 'https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_pos.npy' -P 'data/deepmoji'
!wget 'https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_neg.npy' -P 'data/deepmoji'
!wget 'https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_pos.npy' -P 'data/deepmoji'
!wget 'https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_neg.npy' -P 'data/deepmoji'
```

    --2022-05-07 15:30:11--  https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_pos.npy
    Resolving storage.googleapis.com (storage.googleapis.com)... 108.177.127.128, 172.217.218.128, 142.251.18.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|108.177.127.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 405494864 (387M) [application/octet-stream]
    Saving to: ‘data/deepmoji/pos_pos.npy’
    
    pos_pos.npy         100%[===================>] 386.71M   139MB/s    in 2.8s    
    
    2022-05-07 15:30:14 (139 MB/s) - ‘data/deepmoji/pos_pos.npy’ saved [405494864/405494864]
    
    --2022-05-07 15:30:14--  https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_neg.npy
    Resolving storage.googleapis.com (storage.googleapis.com)... 173.194.69.128, 108.177.127.128, 172.217.218.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|173.194.69.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 405504080 (387M) [application/octet-stream]
    Saving to: ‘data/deepmoji/pos_neg.npy’
    
    pos_neg.npy         100%[===================>] 386.72M   173MB/s    in 2.2s    
    
    2022-05-07 15:30:16 (173 MB/s) - ‘data/deepmoji/pos_neg.npy’ saved [405504080/405504080]
    
    --2022-05-07 15:30:16--  https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_pos.npy
    Resolving storage.googleapis.com (storage.googleapis.com)... 142.251.18.128, 142.250.153.128, 74.125.128.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|142.251.18.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 405494864 (387M) [application/octet-stream]
    Saving to: ‘data/deepmoji/neg_pos.npy’
    
    neg_pos.npy         100%[===================>] 386.71M   136MB/s    in 2.9s    
    
    2022-05-07 15:30:19 (136 MB/s) - ‘data/deepmoji/neg_pos.npy’ saved [405494864/405494864]
    
    --2022-05-07 15:30:19--  https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_neg.npy
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.128.128, 173.194.69.128, 108.177.119.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.128.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 405504080 (387M) [application/octet-stream]
    Saving to: ‘data/deepmoji/neg_neg.npy’
    
    neg_neg.npy         100%[===================>] 386.72M   142MB/s    in 2.7s    
    
    2022-05-07 15:30:22 (142 MB/s) - ‘data/deepmoji/neg_neg.npy’ saved [405504080/405504080]
    
    


```python
fairlib.utils.seed_everything(2022)

import numpy as np
import os

def read_data_file(input_file: str):
    vecs = np.load(input_file)

    np.random.shuffle(vecs)

    return vecs[:40000], vecs[40000:42000], vecs[42000:44000]

in_dir = "data/deepmoji"
out_dir = "data/deepmoji"

os.makedirs(out_dir, exist_ok=True)

for split in ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']:
    train, dev, test = read_data_file(in_dir + '/' + split + '.npy')
    for split_dir, data in zip(['train', 'dev', 'test'], [train, dev, test]):
        os.makedirs(out_dir + '/' + split_dir, exist_ok=True)
        np.save(out_dir + '/' + split_dir + '/' + split + '.npy', data)
```

## Train a Model


```python
Shared_options = {
    # The name of the dataset, corresponding dataloader will be used,
    "dataset":  "Moji",

    # Specifiy the path to the input data
    "data_dir": "data/deepmoji",

    # Device for computing, -1 is the cpu
    "device_id":    -1,

    # The default path for saving experimental results
    "results_dir":  r"results",

    # Will be used for saving experimental results
    "project_dir":  r"dev",

    # We will focusing on TPR GAP, implying the Equalized Odds for binary classification.
    "GAP_metric_name":  "TPR_GAP",

    # The overall performance will be measured as accuracy
    "Performance_metric_name":  "accuracy",

    # Model selections are based on DTO
    "selection_criterion":  "DTO",

    # Default dirs for saving checkpoints
    "checkpoint_dir":   "models",
    "checkpoint_name":  "checkpoint_epoch",

    # Loading experimental results
    "n_jobs":   1,
}
```


```python
args = {
    "dataset":Shared_options["dataset"], 
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    "exp_id":"vanilla",
}

# Init the argument
options = fairlib.BaseOptions()
state = options.get_state(args=args, silence=True)

fairlib.utils.seed_everything(2022)

# Init Model
model = fairlib.networks.get_main_model(state)
```

    INFO:root:Unexpected args: ['-f', '/root/.local/share/jupyter/runtime/kernel-3dad1bfd-dd35-40d7-b985-35feaaff967a.json']
    INFO:root:Logging to ./results/dev/Moji/vanilla/output.log
    

    2022-05-07 15:30:29 [INFO ]  ======================================== 2022-05-07 15:30:29 ========================================
    2022-05-07 15:30:29 [INFO ]  Base directory is ./results/dev/Moji/vanilla
    Loaded data shapes: (99998, 2304), (99998,), (99998,)
    Loaded data shapes: (8000, 2304), (8000,), (8000,)
    Loaded data shapes: (7998, 2304), (7998,), (7998,)
    2022-05-07 15:30:30 [INFO ]  MLP( 
    2022-05-07 15:30:30 [INFO ]    (output_layer): Linear(in_features=300, out_features=2, bias=True)
    2022-05-07 15:30:30 [INFO ]    (AF): Tanh()
    2022-05-07 15:30:30 [INFO ]    (hidden_layers): ModuleList(
    2022-05-07 15:30:30 [INFO ]      (0): Linear(in_features=2304, out_features=300, bias=True)
    2022-05-07 15:30:30 [INFO ]      (1): Tanh()
    2022-05-07 15:30:30 [INFO ]      (2): Linear(in_features=300, out_features=300, bias=True)
    2022-05-07 15:30:30 [INFO ]      (3): Tanh()
    2022-05-07 15:30:30 [INFO ]    )
    2022-05-07 15:30:30 [INFO ]    (criterion): CrossEntropyLoss()
    2022-05-07 15:30:30 [INFO ]  )
    2022-05-07 15:30:30 [INFO ]  Total number of parameters: 782402 
    
    


```python
model.train_self()
```

    2022-05-07 15:30:30 [INFO ]  Epoch:    0 [      0/  99998 ( 0%)]	Loss: 0.6906	 Data Time: 0.02s	Train Time: 0.19s
    2022-05-07 15:30:34 [INFO ]  Epoch:    0 [  51200/  99998 (51%)]	Loss: 0.3926	 Data Time: 0.36s	Train Time: 3.30s
    2022-05-07 15:30:38 [INFO ]  Evaluation at Epoch 0
    2022-05-07 15:30:38 [INFO ]  Validation accuracy: 72.55	macro_fscore: 72.44	micro_fscore: 72.55	TPR_GAP: 40.07	FPR_GAP: 40.07	PPR_GAP: 39.10	
    2022-05-07 15:30:38 [INFO ]  Test accuracy: 71.41	macro_fscore: 71.30	micro_fscore: 71.41	TPR_GAP: 39.01	FPR_GAP: 39.01	PPR_GAP: 37.84	
    2022-05-07 15:30:38 [INFO ]  Epoch:    1 [      0/  99998 ( 0%)]	Loss: 0.4105	 Data Time: 0.01s	Train Time: 0.07s
    2022-05-07 15:30:42 [INFO ]  Epoch:    1 [  51200/  99998 (51%)]	Loss: 0.4156	 Data Time: 0.37s	Train Time: 3.24s
    2022-05-07 15:30:46 [INFO ]  Evaluation at Epoch 1
    2022-05-07 15:30:46 [INFO ]  Validation accuracy: 72.36	macro_fscore: 72.32	micro_fscore: 72.36	TPR_GAP: 39.81	FPR_GAP: 39.81	PPR_GAP: 39.27	
    2022-05-07 15:30:46 [INFO ]  Test accuracy: 71.01	macro_fscore: 70.98	micro_fscore: 71.01	TPR_GAP: 39.40	FPR_GAP: 39.40	PPR_GAP: 38.64	
    2022-05-07 15:30:46 [INFO ]  Epoch:    2 [      0/  99998 ( 0%)]	Loss: 0.3433	 Data Time: 0.01s	Train Time: 0.07s
    2022-05-07 15:30:49 [INFO ]  Epoch:    2 [  51200/  99998 (51%)]	Loss: 0.3734	 Data Time: 0.38s	Train Time: 3.25s
    2022-05-07 15:30:53 [INFO ]  Epochs since last improvement: 1
    2022-05-07 15:30:53 [INFO ]  Evaluation at Epoch 2
    2022-05-07 15:30:53 [INFO ]  Validation accuracy: 72.42	macro_fscore: 72.37	micro_fscore: 72.42	TPR_GAP: 40.91	FPR_GAP: 40.91	PPR_GAP: 40.20	
    2022-05-07 15:30:53 [INFO ]  Test accuracy: 70.98	macro_fscore: 70.93	micro_fscore: 70.98	TPR_GAP: 40.21	FPR_GAP: 40.21	PPR_GAP: 39.39	
    2022-05-07 15:30:53 [INFO ]  Epoch:    3 [      0/  99998 ( 0%)]	Loss: 0.3773	 Data Time: 0.01s	Train Time: 0.06s
    2022-05-07 15:30:57 [INFO ]  Epoch:    3 [  51200/  99998 (51%)]	Loss: 0.3479	 Data Time: 0.38s	Train Time: 3.25s
    2022-05-07 15:31:01 [INFO ]  Epochs since last improvement: 2
    2022-05-07 15:31:01 [INFO ]  Evaluation at Epoch 3
    2022-05-07 15:31:01 [INFO ]  Validation accuracy: 72.09	macro_fscore: 71.92	micro_fscore: 72.09	TPR_GAP: 41.54	FPR_GAP: 41.54	PPR_GAP: 40.17	
    2022-05-07 15:31:01 [INFO ]  Test accuracy: 71.17	macro_fscore: 71.02	micro_fscore: 71.17	TPR_GAP: 40.32	FPR_GAP: 40.32	PPR_GAP: 38.96	
    2022-05-07 15:31:01 [INFO ]  Epoch:    4 [      0/  99998 ( 0%)]	Loss: 0.3839	 Data Time: 0.02s	Train Time: 0.06s
    2022-05-07 15:31:05 [INFO ]  Epoch:    4 [  51200/  99998 (51%)]	Loss: 0.3499	 Data Time: 0.38s	Train Time: 3.28s
    2022-05-07 15:31:11 [INFO ]  Epochs since last improvement: 3
    2022-05-07 15:31:11 [INFO ]  Evaluation at Epoch 4
    2022-05-07 15:31:11 [INFO ]  Validation accuracy: 71.50	macro_fscore: 71.43	micro_fscore: 71.50	TPR_GAP: 42.76	FPR_GAP: 42.76	PPR_GAP: 42.00	
    2022-05-07 15:31:11 [INFO ]  Test accuracy: 70.49	macro_fscore: 70.43	micro_fscore: 70.49	TPR_GAP: 41.37	FPR_GAP: 41.37	PPR_GAP: 40.51	
    2022-05-07 15:31:11 [INFO ]  Epoch:    5 [      0/  99998 ( 0%)]	Loss: 0.3746	 Data Time: 0.03s	Train Time: 0.28s
    2022-05-07 15:31:15 [INFO ]  Epoch:    5 [  51200/  99998 (51%)]	Loss: 0.3748	 Data Time: 0.37s	Train Time: 3.28s
    2022-05-07 15:31:19 [INFO ]  Epochs since last improvement: 4
    2022-05-07 15:31:19 [INFO ]  Evaluation at Epoch 5
    2022-05-07 15:31:19 [INFO ]  Validation accuracy: 72.67	macro_fscore: 72.60	micro_fscore: 72.67	TPR_GAP: 39.17	FPR_GAP: 39.17	PPR_GAP: 38.35	
    2022-05-07 15:31:19 [INFO ]  Test accuracy: 71.69	macro_fscore: 71.62	micro_fscore: 71.69	TPR_GAP: 37.97	FPR_GAP: 37.97	PPR_GAP: 36.91	
    2022-05-07 15:31:19 [INFO ]  Epoch:    6 [      0/  99998 ( 0%)]	Loss: 0.3624	 Data Time: 0.01s	Train Time: 0.06s
    2022-05-07 15:31:23 [INFO ]  Epoch:    6 [  51200/  99998 (51%)]	Loss: 0.3529	 Data Time: 0.38s	Train Time: 3.24s
    2022-05-07 15:31:26 [INFO ]  Epochs since last improvement: 5
    2022-05-07 15:31:27 [INFO ]  Evaluation at Epoch 6
    2022-05-07 15:31:27 [INFO ]  Validation accuracy: 72.70	macro_fscore: 72.62	micro_fscore: 72.70	TPR_GAP: 38.29	FPR_GAP: 38.29	PPR_GAP: 37.50	
    2022-05-07 15:31:27 [INFO ]  Test accuracy: 71.76	macro_fscore: 71.70	micro_fscore: 71.76	TPR_GAP: 37.59	FPR_GAP: 37.59	PPR_GAP: 36.79	
    

By default, fairlib print and save 6 metrics:
- accuracy, macro F1 score, and micro F1 score, which are most commenly used evaluation metrics for performance evaluation.
- rms aggregated TPR, FPR, and PPR GAP scores for fairness assesment. 

## Confusion Matrix Based Metrics


```python
import torch

path = "{results_dir}/{project_dir}/{dataset}/{exp_id}/{checkpoint_dir}/{checkpoint_name}{epoch}.pth.tar"

# Path to the first epoch
path_vanilla_epoch0 = path.format(
    exp_id = "vanilla",
    epoch = "0",
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"],
    dataset=Shared_options["dataset"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
)

epoch_results = torch.load(path_vanilla_epoch0)
# The keys for saved items
print(epoch_results.keys())
```

    dict_keys(['epoch', 'epochs_since_improvement', 'loss', 'valid_confusion_matrices', 'test_confusion_matrices', 'dev_evaluations', 'test_evaluations'])
    

*fairlib* saves confusion matrices for each protected groups as well as the overall confusion matrix. These matrices are stored in a dictionary, indexed with the group id.


```python
epoch_results["valid_confusion_matrices"].keys()
```




    dict_keys(['overall', 0, 1])




```python
epoch_results["valid_confusion_matrices"]["overall"]
```




    array([[2655, 1345],
           [ 851, 3149]])




```python
from fairlib.src.evaluators.evaluator import confusion_matrix_based_scores
```


```python
confusion_matrix_based_scores(epoch_results["valid_confusion_matrices"]["overall"])
```




    {'ACC': array([0.7255, 0.7255]),
     'FDR': array([0.24272676, 0.29928794]),
     'FNR': array([0.33625, 0.21275]),
     'FPR': array([0.21275, 0.33625]),
     'NPV': array([0.70071206, 0.75727324]),
     'PPR': array([0.43825, 0.56175]),
     'PPV': array([0.75727324, 0.70071206]),
     'TNR': array([0.78725, 0.66375]),
     'TPR': array([0.66375, 0.78725])}




```python
from fairlib.src.evaluators.evaluator import power_mean
```


```python
numbers = np.array([1,2,3,4,5])
# generalized mean aggregation
[
 power_mean(numbers, p=100), # Max
 power_mean(numbers, p=2), # Root Mean Square
 power_mean(numbers, p=1), # Arithmetic Mean
 power_mean(numbers, p=-100), # Min
]
```




    [5, 3.3166247903554, 3.0, 1]



- Max Violation
- RMS GAP 
- Max Min Fairness


```python
from fairlib.src.evaluators.evaluator import Aggregation_GAP
```


```python
# Confusion matrices of the vanilla model's first epoch over the vailidation dataset
confusion_matrices = epoch_results["valid_confusion_matrices"]

# all_scores = dict()
all_scores = {}

# Overall evaluation
all_scores["overall"] = confusion_matrix_based_scores(confusion_matrices["overall"])

# Group scores
distinct_groups = [0,1] # binary protected groups, AAE verse SAE
for gid in distinct_groups:
    group_confusion_matrix = confusion_matrices[gid]
    all_scores[gid] = confusion_matrix_based_scores(group_confusion_matrix)
```


```python
Aggregation_GAP(
    distinct_groups=[0,1], 
    all_scores=all_scores, 

    # Take the absolute different if None, 
    # using generalized mean aggregation if not None.
    group_agg_power = -10,
    
    # RMS aggregation by default
    class_agg_power=2, 
    metric="TPR")
```




    0.2003354978462722



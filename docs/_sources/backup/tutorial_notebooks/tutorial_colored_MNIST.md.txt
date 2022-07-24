# Using *fairlib* with Image Inputs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HanXudong/fairlib/blob/main/tutorial/colored_MNIST.ipynb)

In this tutorial we will:
- Show how to add a model for image classification
- Show how to add a dataloader with image preprocessing

We will be using the colored MNIST dataset, please see the [origianl](https://colab.research.google.com/github/reiinakano/invariant-risk-minimization/blob/master/invariant_risk_minimization_colored_mnist.ipynb#scrollTo=knP-xNzavgAb&uniqifier=1) demo for more details.

## Installation

Again, the first step will be installing our libarary


```python
!pip install fairlib
```

    Collecting fairlib
      Downloading fairlib-0.0.1-py3-none-any.whl (61 kB)
    [K     |████████████████████████████████| 61 kB 3.6 MB/s 
    [?25hCollecting transformers
      Downloading transformers-4.18.0-py3-none-any.whl (4.0 MB)
    [K     |████████████████████████████████| 4.0 MB 14.8 MB/s 
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from fairlib) (4.63.0)
    Requirement already satisfied: docopt in /usr/local/lib/python3.7/dist-packages (from fairlib) (0.6.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.0.2)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.7/dist-packages (from fairlib) (0.11.2)
    Requirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.10.0+cu111)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from fairlib) (3.13)
    Collecting pickle5
      Downloading pickle5-0.0.12-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (256 kB)
    [K     |████████████████████████████████| 256 kB 64.7 MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.21.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from fairlib) (3.2.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.3.5)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (3.0.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (1.4.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (0.11.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (2.8.2)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->fairlib) (3.10.0.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib->fairlib) (1.15.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->fairlib) (2018.9)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (1.1.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (3.1.0)
    Requirement already satisfied: scipy>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (1.4.1)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (21.3)
    Collecting huggingface-hub<1.0,>=0.1.0
      Downloading huggingface_hub-0.5.1-py3-none-any.whl (77 kB)
    [K     |████████████████████████████████| 77 kB 5.2 MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (2019.12.20)
    Collecting PyYAML
      Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
    [K     |████████████████████████████████| 596 kB 50.9 MB/s 
    [?25hRequirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (4.11.3)
    Collecting tokenizers!=0.11.3,<0.13,>=0.11.1
      Downloading tokenizers-0.11.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.5 MB)
    [K     |████████████████████████████████| 6.5 MB 48.3 MB/s 
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (3.6.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (2.23.0)
    Collecting sacremoses
      Downloading sacremoses-0.0.49-py3-none-any.whl (895 kB)
    [K     |████████████████████████████████| 895 kB 47.9 MB/s 
    [?25hRequirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers->fairlib) (3.7.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (2021.10.8)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (2.10)
    Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers->fairlib) (7.1.2)
    Installing collected packages: PyYAML, tokenizers, sacremoses, huggingface-hub, transformers, pickle5, fairlib
      Attempting uninstall: PyYAML
        Found existing installation: PyYAML 3.13
        Uninstalling PyYAML-3.13:
          Successfully uninstalled PyYAML-3.13
    Successfully installed PyYAML-6.0 fairlib-0.0.1 huggingface-hub-0.5.1 pickle5-0.0.12 sacremoses-0.0.49 tokenizers-0.11.6 transformers-4.18.0
    


```python
import fairlib
```

## Prepare Data

```python
import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
```


```python
def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([
            arr,
            np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([
            np.zeros((h, w, 1), dtype=dtype),
            arr,
            np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr
```


```python
root = "./data"
```


```python
train_mnist = datasets.mnist.MNIST(root=root, train=True, download=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw
    
    


```python
len(train_mnist)
```




    60000




```python
fig, ax = plt.subplots(1,3)
ax[0].imshow(train_mnist[0][0]) # Original
ax[1].imshow(color_grayscale_arr(np.array(train_mnist[0][0]))) # Red
ax[2].imshow(color_grayscale_arr(np.array(train_mnist[0][0]), False)) # Green
```




    <matplotlib.image.AxesImage at 0x7f1b9516cb10>




    
![png](tutorial_colored_MNIST_files/tutorial_colored_MNIST_8_1.png)
    



```python
test_mnist = datasets.mnist.MNIST(root=root, train=False, download=True)
```


```python
fig, ax = plt.subplots(1,3)
ax[0].imshow(test_mnist[0][0]) # Original
ax[1].imshow(color_grayscale_arr(np.array(test_mnist[0][0]))) # Red
ax[2].imshow(color_grayscale_arr(np.array(test_mnist[0][0]), False)) # Green
```




    <matplotlib.image.AxesImage at 0x7f1b950ad650>




    
![png](tutorial_colored_MNIST_files/tutorial_colored_MNIST_10_1.png)
    



```python
def process_colored_MNIST(loaded_data, ratio=0.2):
    im_list, target_class, protected_class = [], [], []
    # Iterate over all images
    for idx, (im, label) in enumerate(loaded_data):
        if idx % 5000 == 0:
            print(f'Converting image {idx}/{len(loaded_data)}')
        im_array = np.array(im)

        # Assign a binary label y to the image based on the digit
        if label < 5:
            red_prob = ratio
        else:
            red_prob = 1-ratio
        
        color_red = (np.random.uniform() < red_prob)

        colored_array = color_grayscale_arr(im_array, red=color_red)

        im_list.append(Image.fromarray(colored_array))
        target_class.append(int(label))
        protected_class.append(int(color_red))
    
    return (im_list, target_class, protected_class)

```


```python
num_train = len(train_mnist)
indices = list(range(num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[10000:], indices[:10000]
```


```python
colored_MNIST_train = process_colored_MNIST(torch.utils.data.Subset(train_mnist, train_idx), ratio = 0.8)
colored_MNIST_dev = process_colored_MNIST(torch.utils.data.Subset(train_mnist, valid_idx), ratio = 0.5)
colored_MNIST_test = process_colored_MNIST(test_mnist, ratio = 0.5)
```

    Converting image 0/50000
    Converting image 5000/50000
    Converting image 10000/50000
    Converting image 15000/50000
    Converting image 20000/50000
    Converting image 25000/50000
    Converting image 30000/50000
    Converting image 35000/50000
    Converting image 40000/50000
    Converting image 45000/50000
    Converting image 0/10000
    Converting image 5000/10000
    Converting image 0/10000
    Converting image 5000/10000
    


```python
torch.save(colored_MNIST_train, os.path.join("./data", "colored_MNIST_train.pt"))
torch.save(colored_MNIST_dev, os.path.join("./data", "colored_MNIST_dev.pt"))
torch.save(colored_MNIST_test, os.path.join("./data", "colored_MNIST_test.pt"))
```


## Train a Vanilla Model 

```python
from fairlib import networks, BaseOptions, dataloaders
```


```python
Shared_options = {
    # The name of the dataset, correponding dataloader will be used,
    "dataset":  "MNIST",

    # Specifiy the path to the input data
    "data_dir": "./data",

    # Device for computing, -1 is the cpu
    "device_id": -1,

    # The default path for saving experimental results
    "results_dir":  r"results",

    # The same as the dataset
    "project_dir":  r"dev",

    # We will focusing on TPR GAP, implying the Equalized Odds for binay classification.
    "GAP_metric_name":  "TPR_GAP",

    # The overall performance will be measured as accuracy
    "Performance_metric_name":  "accuracy",

    # Model selections are based on DTO
    "selection_criterion":  "DTO",

    # Default dirs for saving checkpoints
    "checkpoint_dir":   "models",
    "checkpoint_name":  "checkpoint_epoch",


    "n_jobs":   1,
}
```


```python
class ConvNet(networks.utils.BaseModel):

    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.args = args
        
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.classifier = networks.classifier.MLP(args)

        self.init_for_training()

    def forward(self, input_data, group_label = None):
        x = input_data
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        return self.classifier(x, group_label)
    
    def hidden(self, input_data, group_label = None):
        x = input_data
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        return self.classifier.hidden(x, group_label)
```


```python
args = {
    "dataset":Shared_options["dataset"], 
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    "exp_id":"vanilla",

    "emb_size": 4*4*50,
    "num_classes": 10,
    "num_groups": 2,
}

# Init the argument
options = BaseOptions()
state = options.get_state(args=args, silence=True)
```

    INFO:root:Unexpected args: ['-f', '/root/.local/share/jupyter/runtime/kernel-9c3335e9-652e-4f42-97e0-8b86721e4921.json']
    INFO:root:Logging to ./results/dev/MNIST/vanilla/output.log
    

    2022-04-08 15:15:04 [INFO ]  ======================================== 2022-04-08 15:15:04 ========================================
    2022-04-08 15:15:04 [INFO ]  Base directory is ./results/dev/MNIST/vanilla
    Not implemented
    2022-04-08 15:15:04 [INFO ]  dataloaders need to be initialized!
    


```python
class CustomizedDataset(dataloaders.utils.BaseDataset):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))])

    def load_data(self):

        self.data_dir = os.path.join(self.args.data_dir, "colored_MNIST_{}.pt".format(self.split))

        data = torch.load(self.data_dir)

        self.X = [self.transform(_img) for _img in data[0]]
        self.y = data[1]
        self.protected_label = data[2]
```


```python
customized_train_data = CustomizedDataset(args=state, split="train")
customized_dev_data = CustomizedDataset(args=state, split="dev")
customized_test_data = CustomizedDataset(args=state, split="test")

# DataLoader Parameters
tran_dataloader_params = {
        'batch_size': state.batch_size,
        'shuffle': True,
        'num_workers': state.num_workers}

eval_dataloader_params = {
        'batch_size': state.test_batch_size,
        'shuffle': False,
        'num_workers': state.num_workers}

# init dataloader
customized_training_generator = torch.utils.data.DataLoader(customized_train_data, **tran_dataloader_params)
customized_validation_generator = torch.utils.data.DataLoader(customized_dev_data, **eval_dataloader_params)
customized_test_generator = torch.utils.data.DataLoader(customized_test_data, **eval_dataloader_params)
```

    /usr/local/lib/python3.7/dist-packages/fairlib/src/dataloaders/utils.py:163: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      self.X = np.array(self.X)
    

    Loaded data shapes: (50000,), (50000,), (50000,)
    Loaded data shapes: (10000,), (10000,), (10000,)
    Loaded data shapes: (10000,), (10000,), (10000,)
    


```python
model = ConvNet(state)
```

    2022-04-08 15:15:28 [INFO ]  MLP( 
    2022-04-08 15:15:28 [INFO ]    (output_layer): Linear(in_features=300, out_features=10, bias=True)
    2022-04-08 15:15:28 [INFO ]    (AF): Tanh()
    2022-04-08 15:15:28 [INFO ]    (hidden_layers): ModuleList(
    2022-04-08 15:15:28 [INFO ]      (0): Linear(in_features=800, out_features=300, bias=True)
    2022-04-08 15:15:28 [INFO ]      (1): Tanh()
    2022-04-08 15:15:28 [INFO ]      (2): Linear(in_features=300, out_features=300, bias=True)
    2022-04-08 15:15:28 [INFO ]      (3): Tanh()
    2022-04-08 15:15:28 [INFO ]    )
    2022-04-08 15:15:28 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:15:28 [INFO ]  )
    2022-04-08 15:15:28 [INFO ]  Total number of parameters: 333610 
    
    2022-04-08 15:15:28 [INFO ]  ConvNet( 
    2022-04-08 15:15:28 [INFO ]    (conv1): Conv2d(3, 20, kernel_size=(5, 5), stride=(1, 1))
    2022-04-08 15:15:28 [INFO ]    (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
    2022-04-08 15:15:28 [INFO ]    (classifier): MLP(
    2022-04-08 15:15:28 [INFO ]      (output_layer): Linear(in_features=300, out_features=10, bias=True)
    2022-04-08 15:15:28 [INFO ]      (AF): Tanh()
    2022-04-08 15:15:28 [INFO ]      (hidden_layers): ModuleList(
    2022-04-08 15:15:28 [INFO ]        (0): Linear(in_features=800, out_features=300, bias=True)
    2022-04-08 15:15:28 [INFO ]        (1): Tanh()
    2022-04-08 15:15:28 [INFO ]        (2): Linear(in_features=300, out_features=300, bias=True)
    2022-04-08 15:15:28 [INFO ]        (3): Tanh()
    2022-04-08 15:15:28 [INFO ]      )
    2022-04-08 15:15:28 [INFO ]      (criterion): CrossEntropyLoss()
    2022-04-08 15:15:28 [INFO ]    )
    2022-04-08 15:15:28 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:15:28 [INFO ]  )
    2022-04-08 15:15:28 [INFO ]  Total number of parameters: 360180 
    
    


```python
model.train_self(
    train_generator = customized_training_generator,
    dev_generator = customized_validation_generator,
    test_generator = customized_test_generator,
)
```

    2022-04-08 15:15:31 [INFO ]  Epoch:    0 [      0/  50000 ( 0%)]	Loss: 2.3092	 Data Time: 0.02s	Train Time: 1.06s
    2022-04-08 15:16:09 [INFO ]  Evaluation at Epoch 0
    2022-04-08 15:16:09 [INFO ]  Validation accuracy: 95.66	macro_fscore: 95.62	micro_fscore: 95.66	TPR_GAP: 4.20	FPR_GAP: 0.42	PPR_GAP: 0.89	
    2022-04-08 15:16:09 [INFO ]  Test accuracy: 95.89	macro_fscore: 95.86	micro_fscore: 95.89	TPR_GAP: 4.93	FPR_GAP: 0.43	PPR_GAP: 1.01	
    2022-04-08 15:16:10 [INFO ]  Epoch:    1 [      0/  50000 ( 0%)]	Loss: 0.1109	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:16:48 [INFO ]  Evaluation at Epoch 1
    2022-04-08 15:16:48 [INFO ]  Validation accuracy: 97.50	macro_fscore: 97.49	micro_fscore: 97.50	TPR_GAP: 2.36	FPR_GAP: 0.29	PPR_GAP: 0.70	
    2022-04-08 15:16:48 [INFO ]  Test accuracy: 97.70	macro_fscore: 97.70	micro_fscore: 97.70	TPR_GAP: 3.01	FPR_GAP: 0.35	PPR_GAP: 0.75	
    2022-04-08 15:16:48 [INFO ]  Epoch:    2 [      0/  50000 ( 0%)]	Loss: 0.0479	 Data Time: 0.01s	Train Time: 0.63s
    2022-04-08 15:17:24 [INFO ]  Evaluation at Epoch 2
    2022-04-08 15:17:24 [INFO ]  Validation accuracy: 98.05	macro_fscore: 98.03	micro_fscore: 98.05	TPR_GAP: 1.54	FPR_GAP: 0.20	PPR_GAP: 0.58	
    2022-04-08 15:17:24 [INFO ]  Test accuracy: 98.44	macro_fscore: 98.42	micro_fscore: 98.44	TPR_GAP: 1.42	FPR_GAP: 0.15	PPR_GAP: 0.52	
    2022-04-08 15:17:25 [INFO ]  Epoch:    3 [      0/  50000 ( 0%)]	Loss: 0.0367	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:18:01 [INFO ]  Evaluation at Epoch 3
    2022-04-08 15:18:01 [INFO ]  Validation accuracy: 98.15	macro_fscore: 98.15	micro_fscore: 98.15	TPR_GAP: 2.43	FPR_GAP: 0.21	PPR_GAP: 0.73	
    2022-04-08 15:18:01 [INFO ]  Test accuracy: 98.32	macro_fscore: 98.31	micro_fscore: 98.32	TPR_GAP: 2.52	FPR_GAP: 0.23	PPR_GAP: 0.71	
    2022-04-08 15:18:02 [INFO ]  Epoch:    4 [      0/  50000 ( 0%)]	Loss: 0.0225	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:18:38 [INFO ]  Evaluation at Epoch 4
    2022-04-08 15:18:38 [INFO ]  Validation accuracy: 98.25	macro_fscore: 98.24	micro_fscore: 98.25	TPR_GAP: 1.92	FPR_GAP: 0.19	PPR_GAP: 0.57	
    2022-04-08 15:18:38 [INFO ]  Test accuracy: 98.48	macro_fscore: 98.48	micro_fscore: 98.48	TPR_GAP: 2.00	FPR_GAP: 0.20	PPR_GAP: 0.65	
    2022-04-08 15:18:38 [INFO ]  Epoch:    5 [      0/  50000 ( 0%)]	Loss: 0.0089	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:19:14 [INFO ]  Evaluation at Epoch 5
    2022-04-08 15:19:14 [INFO ]  Validation accuracy: 98.54	macro_fscore: 98.53	micro_fscore: 98.54	TPR_GAP: 1.22	FPR_GAP: 0.15	PPR_GAP: 0.58	
    2022-04-08 15:19:14 [INFO ]  Test accuracy: 98.98	macro_fscore: 98.98	micro_fscore: 98.98	TPR_GAP: 1.70	FPR_GAP: 0.14	PPR_GAP: 0.62	
    2022-04-08 15:19:15 [INFO ]  Epoch:    6 [      0/  50000 ( 0%)]	Loss: 0.0104	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:19:51 [INFO ]  Evaluation at Epoch 6
    2022-04-08 15:19:51 [INFO ]  Validation accuracy: 98.66	macro_fscore: 98.65	micro_fscore: 98.66	TPR_GAP: 1.23	FPR_GAP: 0.12	PPR_GAP: 0.54	
    2022-04-08 15:19:51 [INFO ]  Test accuracy: 98.83	macro_fscore: 98.83	micro_fscore: 98.83	TPR_GAP: 1.27	FPR_GAP: 0.18	PPR_GAP: 0.60	
    2022-04-08 15:19:52 [INFO ]  Epoch:    7 [      0/  50000 ( 0%)]	Loss: 0.0110	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:20:26 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:20:29 [INFO ]  Evaluation at Epoch 7
    2022-04-08 15:20:29 [INFO ]  Validation accuracy: 98.40	macro_fscore: 98.39	micro_fscore: 98.40	TPR_GAP: 1.66	FPR_GAP: 0.16	PPR_GAP: 0.63	
    2022-04-08 15:20:29 [INFO ]  Test accuracy: 98.91	macro_fscore: 98.90	micro_fscore: 98.91	TPR_GAP: 1.55	FPR_GAP: 0.13	PPR_GAP: 0.57	
    2022-04-08 15:20:30 [INFO ]  Epoch:    8 [      0/  50000 ( 0%)]	Loss: 0.0040	 Data Time: 0.01s	Train Time: 0.70s
    2022-04-08 15:21:03 [INFO ]  Epochs since last improvement: 2
    2022-04-08 15:21:06 [INFO ]  Evaluation at Epoch 8
    2022-04-08 15:21:06 [INFO ]  Validation accuracy: 98.44	macro_fscore: 98.43	micro_fscore: 98.44	TPR_GAP: 1.91	FPR_GAP: 0.18	PPR_GAP: 0.58	
    2022-04-08 15:21:06 [INFO ]  Test accuracy: 98.80	macro_fscore: 98.79	micro_fscore: 98.80	TPR_GAP: 2.21	FPR_GAP: 0.22	PPR_GAP: 0.60	
    2022-04-08 15:21:06 [INFO ]  Epoch:    9 [      0/  50000 ( 0%)]	Loss: 0.0037	 Data Time: 0.01s	Train Time: 0.63s
    2022-04-08 15:21:40 [INFO ]  Epochs since last improvement: 3
    2022-04-08 15:21:42 [INFO ]  Evaluation at Epoch 9
    2022-04-08 15:21:42 [INFO ]  Validation accuracy: 98.51	macro_fscore: 98.50	micro_fscore: 98.51	TPR_GAP: 1.01	FPR_GAP: 0.14	PPR_GAP: 0.57	
    2022-04-08 15:21:42 [INFO ]  Test accuracy: 98.99	macro_fscore: 98.98	micro_fscore: 98.99	TPR_GAP: 1.03	FPR_GAP: 0.14	PPR_GAP: 0.57	
    2022-04-08 15:21:43 [INFO ]  Epoch:   10 [      0/  50000 ( 0%)]	Loss: 0.0045	 Data Time: 0.01s	Train Time: 0.63s
    2022-04-08 15:22:19 [INFO ]  Evaluation at Epoch 10
    2022-04-08 15:22:19 [INFO ]  Validation accuracy: 98.75	macro_fscore: 98.74	micro_fscore: 98.75	TPR_GAP: 1.02	FPR_GAP: 0.11	PPR_GAP: 0.59	
    2022-04-08 15:22:19 [INFO ]  Test accuracy: 99.02	macro_fscore: 99.01	micro_fscore: 99.02	TPR_GAP: 1.08	FPR_GAP: 0.12	PPR_GAP: 0.54	
    2022-04-08 15:22:20 [INFO ]  Epoch:   11 [      0/  50000 ( 0%)]	Loss: 0.0023	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:22:53 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:22:56 [INFO ]  Evaluation at Epoch 11
    2022-04-08 15:22:56 [INFO ]  Validation accuracy: 98.66	macro_fscore: 98.65	micro_fscore: 98.66	TPR_GAP: 1.38	FPR_GAP: 0.14	PPR_GAP: 0.58	
    2022-04-08 15:22:56 [INFO ]  Test accuracy: 99.03	macro_fscore: 99.03	micro_fscore: 99.03	TPR_GAP: 0.91	FPR_GAP: 0.11	PPR_GAP: 0.50	
    2022-04-08 15:22:56 [INFO ]  Epoch:   12 [      0/  50000 ( 0%)]	Loss: 0.0019	 Data Time: 0.01s	Train Time: 0.63s
    2022-04-08 15:23:30 [INFO ]  Epochs since last improvement: 2
    2022-04-08 15:23:32 [INFO ]  Evaluation at Epoch 12
    2022-04-08 15:23:32 [INFO ]  Validation accuracy: 98.71	macro_fscore: 98.70	micro_fscore: 98.71	TPR_GAP: 1.19	FPR_GAP: 0.10	PPR_GAP: 0.57	
    2022-04-08 15:23:32 [INFO ]  Test accuracy: 99.08	macro_fscore: 99.07	micro_fscore: 99.08	TPR_GAP: 0.94	FPR_GAP: 0.10	PPR_GAP: 0.53	
    2022-04-08 15:23:33 [INFO ]  Epoch:   13 [      0/  50000 ( 0%)]	Loss: 0.0011	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:24:07 [INFO ]  Epochs since last improvement: 3
    2022-04-08 15:24:09 [INFO ]  Evaluation at Epoch 13
    2022-04-08 15:24:09 [INFO ]  Validation accuracy: 98.79	macro_fscore: 98.78	micro_fscore: 98.79	TPR_GAP: 1.05	FPR_GAP: 0.12	PPR_GAP: 0.58	
    2022-04-08 15:24:09 [INFO ]  Test accuracy: 99.16	macro_fscore: 99.15	micro_fscore: 99.16	TPR_GAP: 0.98	FPR_GAP: 0.10	PPR_GAP: 0.55	
    2022-04-08 15:24:10 [INFO ]  Epoch:   14 [      0/  50000 ( 0%)]	Loss: 0.0004	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:24:43 [INFO ]  Epochs since last improvement: 4
    2022-04-08 15:24:46 [INFO ]  Evaluation at Epoch 14
    2022-04-08 15:24:46 [INFO ]  Validation accuracy: 98.78	macro_fscore: 98.77	micro_fscore: 98.78	TPR_GAP: 1.25	FPR_GAP: 0.12	PPR_GAP: 0.57	
    2022-04-08 15:24:46 [INFO ]  Test accuracy: 99.22	macro_fscore: 99.22	micro_fscore: 99.22	TPR_GAP: 0.83	FPR_GAP: 0.10	PPR_GAP: 0.51	
    2022-04-08 15:24:46 [INFO ]  Epoch:   15 [      0/  50000 ( 0%)]	Loss: 0.0002	 Data Time: 0.01s	Train Time: 0.64s
    2022-04-08 15:25:20 [INFO ]  Epochs since last improvement: 5
    2022-04-08 15:25:22 [INFO ]  Evaluation at Epoch 15
    2022-04-08 15:25:22 [INFO ]  Validation accuracy: 98.78	macro_fscore: 98.77	micro_fscore: 98.78	TPR_GAP: 1.17	FPR_GAP: 0.12	PPR_GAP: 0.59	
    2022-04-08 15:25:22 [INFO ]  Test accuracy: 99.20	macro_fscore: 99.20	micro_fscore: 99.20	TPR_GAP: 0.87	FPR_GAP: 0.11	PPR_GAP: 0.54	
    

## Bias Mitigation

```python
debiasing_args = {
    "dataset":Shared_options["dataset"], 
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    "exp_id":"BT_Adv",

    "emb_size": 4*4*50,
    "num_classes": 10,
    "num_groups": 2,

    # Perform adversarial training if True
    "adv_debiasing":True,

    # Specify the hyperparameters for Balanced Training
    "BT":"Resampling",
    "BTObj":"EO",
}

# Init the argument
debias_options = BaseOptions()
debias_state = debias_options.get_state(args=debiasing_args, silence=True)

customized_train_data = CustomizedDataset(args=debias_state, split="train")
customized_dev_data = CustomizedDataset(args=debias_state, split="dev")
customized_test_data = CustomizedDataset(args=debias_state, split="test")

# DataLoader Parameters
tran_dataloader_params = {
        'batch_size': state.batch_size,
        'shuffle': True,
        'num_workers': state.num_workers}

eval_dataloader_params = {
        'batch_size': state.test_batch_size,
        'shuffle': False,
        'num_workers': state.num_workers}

# init dataloader
customized_training_generator = torch.utils.data.DataLoader(customized_train_data, **tran_dataloader_params)
customized_validation_generator = torch.utils.data.DataLoader(customized_dev_data, **eval_dataloader_params)
customized_test_generator = torch.utils.data.DataLoader(customized_test_data, **eval_dataloader_params)

debias_model = ConvNet(debias_state)
```

    2022-04-08 15:25:37 [INFO ]  Unexpected args: ['-f', '/root/.local/share/jupyter/runtime/kernel-9c3335e9-652e-4f42-97e0-8b86721e4921.json']
    2022-04-08 15:25:37 [INFO ]  Logging to ./results/dev/MNIST/BT_Adv/output.log
    2022-04-08 15:25:37 [INFO ]  ======================================== 2022-04-08 15:25:37 ========================================
    2022-04-08 15:25:37 [INFO ]  Base directory is ./results/dev/MNIST/BT_Adv
    Not implemented
    2022-04-08 15:25:37 [INFO ]  dataloaders need to be initialized!
    2022-04-08 15:25:37 [INFO ]  SubDiscriminator( 
    2022-04-08 15:25:37 [INFO ]    (grad_rev): GradientReversal()
    2022-04-08 15:25:37 [INFO ]    (output_layer): Linear(in_features=300, out_features=2, bias=True)
    2022-04-08 15:25:37 [INFO ]    (AF): ReLU()
    2022-04-08 15:25:37 [INFO ]    (hidden_layers): ModuleList(
    2022-04-08 15:25:37 [INFO ]      (0): Linear(in_features=300, out_features=300, bias=True)
    2022-04-08 15:25:37 [INFO ]      (1): ReLU()
    2022-04-08 15:25:37 [INFO ]      (2): Linear(in_features=300, out_features=300, bias=True)
    2022-04-08 15:25:37 [INFO ]      (3): ReLU()
    2022-04-08 15:25:37 [INFO ]    )
    2022-04-08 15:25:37 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:25:37 [INFO ]  )
    2022-04-08 15:25:37 [INFO ]  Total number of parameters: 181202 
    
    2022-04-08 15:25:38 [INFO ]  Discriminator built!
    

    /usr/local/lib/python3.7/dist-packages/fairlib/src/dataloaders/utils.py:163: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      self.X = np.array(self.X)
    /usr/local/lib/python3.7/dist-packages/fairlib/src/dataloaders/utils.py:211: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      self.X = np.array(X)
    

    Loaded data shapes: (20018,), (20018,), (20018,)
    Loaded data shapes: (9794,), (9794,), (9794,)
    Loaded data shapes: (9798,), (9798,), (9798,)
    2022-04-08 15:25:51 [INFO ]  MLP( 
    2022-04-08 15:25:51 [INFO ]    (output_layer): Linear(in_features=300, out_features=10, bias=True)
    2022-04-08 15:25:51 [INFO ]    (AF): Tanh()
    2022-04-08 15:25:51 [INFO ]    (hidden_layers): ModuleList(
    2022-04-08 15:25:51 [INFO ]      (0): Linear(in_features=800, out_features=300, bias=True)
    2022-04-08 15:25:51 [INFO ]      (1): Tanh()
    2022-04-08 15:25:51 [INFO ]      (2): Linear(in_features=300, out_features=300, bias=True)
    2022-04-08 15:25:51 [INFO ]      (3): Tanh()
    2022-04-08 15:25:51 [INFO ]    )
    2022-04-08 15:25:51 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:25:51 [INFO ]  )
    2022-04-08 15:25:51 [INFO ]  Total number of parameters: 333610 
    
    2022-04-08 15:25:51 [INFO ]  ConvNet( 
    2022-04-08 15:25:51 [INFO ]    (conv1): Conv2d(3, 20, kernel_size=(5, 5), stride=(1, 1))
    2022-04-08 15:25:51 [INFO ]    (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
    2022-04-08 15:25:51 [INFO ]    (classifier): MLP(
    2022-04-08 15:25:51 [INFO ]      (output_layer): Linear(in_features=300, out_features=10, bias=True)
    2022-04-08 15:25:51 [INFO ]      (AF): Tanh()
    2022-04-08 15:25:51 [INFO ]      (hidden_layers): ModuleList(
    2022-04-08 15:25:51 [INFO ]        (0): Linear(in_features=800, out_features=300, bias=True)
    2022-04-08 15:25:51 [INFO ]        (1): Tanh()
    2022-04-08 15:25:51 [INFO ]        (2): Linear(in_features=300, out_features=300, bias=True)
    2022-04-08 15:25:51 [INFO ]        (3): Tanh()
    2022-04-08 15:25:51 [INFO ]      )
    2022-04-08 15:25:51 [INFO ]      (criterion): CrossEntropyLoss()
    2022-04-08 15:25:51 [INFO ]    )
    2022-04-08 15:25:51 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:25:51 [INFO ]  )
    2022-04-08 15:25:51 [INFO ]  Total number of parameters: 360180 
    
    


```python
debias_model.train_self(
    train_generator = customized_training_generator,
    dev_generator = customized_validation_generator,
    test_generator = customized_test_generator,
)
```

    2022-04-08 15:26:01 [INFO ]  Epoch:    0 [      0/  20018 ( 0%)]	Loss: 1.6324	 Data Time: 0.02s	Train Time: 2.12s
    2022-04-08 15:26:40 [INFO ]  Evaluation at Epoch 0
    2022-04-08 15:26:40 [INFO ]  Validation accuracy: 91.53	macro_fscore: 91.46	micro_fscore: 91.53	TPR_GAP: 5.13	FPR_GAP: 0.75	PPR_GAP: 1.08	
    2022-04-08 15:26:40 [INFO ]  Test accuracy: 92.26	macro_fscore: 92.18	micro_fscore: 92.26	TPR_GAP: 4.50	FPR_GAP: 0.50	PPR_GAP: 0.83	
    2022-04-08 15:26:42 [INFO ]  Epoch:    1 [      0/  20018 ( 0%)]	Loss: -0.4085	 Data Time: 0.01s	Train Time: 1.82s
    2022-04-08 15:27:21 [INFO ]  Evaluation at Epoch 1
    2022-04-08 15:27:21 [INFO ]  Validation accuracy: 94.82	macro_fscore: 94.77	micro_fscore: 94.82	TPR_GAP: 4.61	FPR_GAP: 0.52	PPR_GAP: 0.69	
    2022-04-08 15:27:21 [INFO ]  Test accuracy: 95.24	macro_fscore: 95.18	micro_fscore: 95.24	TPR_GAP: 3.36	FPR_GAP: 0.49	PPR_GAP: 0.63	
    2022-04-08 15:27:23 [INFO ]  Epoch:    2 [      0/  20018 ( 0%)]	Loss: -0.5775	 Data Time: 0.01s	Train Time: 1.79s
    2022-04-08 15:28:02 [INFO ]  Evaluation at Epoch 2
    2022-04-08 15:28:02 [INFO ]  Validation accuracy: 96.91	macro_fscore: 96.89	micro_fscore: 96.91	TPR_GAP: 1.45	FPR_GAP: 0.11	PPR_GAP: 0.21	
    2022-04-08 15:28:02 [INFO ]  Test accuracy: 96.94	macro_fscore: 96.92	micro_fscore: 96.94	TPR_GAP: 1.21	FPR_GAP: 0.11	PPR_GAP: 0.20	
    2022-04-08 15:28:03 [INFO ]  Epoch:    3 [      0/  20018 ( 0%)]	Loss: -0.5939	 Data Time: 0.01s	Train Time: 1.74s
    2022-04-08 15:28:42 [INFO ]  Evaluation at Epoch 3
    2022-04-08 15:28:42 [INFO ]  Validation accuracy: 97.77	macro_fscore: 97.76	micro_fscore: 97.77	TPR_GAP: 1.13	FPR_GAP: 0.13	PPR_GAP: 0.20	
    2022-04-08 15:28:42 [INFO ]  Test accuracy: 98.15	macro_fscore: 98.15	micro_fscore: 98.15	TPR_GAP: 1.04	FPR_GAP: 0.13	PPR_GAP: 0.19	
    2022-04-08 15:28:43 [INFO ]  Epoch:    4 [      0/  20018 ( 0%)]	Loss: -0.6504	 Data Time: 0.01s	Train Time: 1.73s
    2022-04-08 15:29:22 [INFO ]  Evaluation at Epoch 4
    2022-04-08 15:29:22 [INFO ]  Validation accuracy: 97.92	macro_fscore: 97.91	micro_fscore: 97.92	TPR_GAP: 0.85	FPR_GAP: 0.10	PPR_GAP: 0.14	
    2022-04-08 15:29:22 [INFO ]  Test accuracy: 98.40	macro_fscore: 98.39	micro_fscore: 98.40	TPR_GAP: 1.10	FPR_GAP: 0.13	PPR_GAP: 0.19	
    2022-04-08 15:29:24 [INFO ]  Epoch:    5 [      0/  20018 ( 0%)]	Loss: -0.6614	 Data Time: 0.01s	Train Time: 1.76s
    2022-04-08 15:30:03 [INFO ]  Evaluation at Epoch 5
    2022-04-08 15:30:03 [INFO ]  Validation accuracy: 98.09	macro_fscore: 98.08	micro_fscore: 98.09	TPR_GAP: 0.79	FPR_GAP: 0.13	PPR_GAP: 0.18	
    2022-04-08 15:30:03 [INFO ]  Test accuracy: 98.34	macro_fscore: 98.33	micro_fscore: 98.34	TPR_GAP: 1.15	FPR_GAP: 0.07	PPR_GAP: 0.14	
    2022-04-08 15:30:04 [INFO ]  Epoch:    6 [      0/  20018 ( 0%)]	Loss: -0.6657	 Data Time: 0.01s	Train Time: 1.73s
    2022-04-08 15:30:45 [INFO ]  Evaluation at Epoch 6
    2022-04-08 15:30:45 [INFO ]  Validation accuracy: 98.25	macro_fscore: 98.24	micro_fscore: 98.25	TPR_GAP: 0.72	FPR_GAP: 0.09	PPR_GAP: 0.11	
    2022-04-08 15:30:45 [INFO ]  Test accuracy: 98.63	macro_fscore: 98.62	micro_fscore: 98.63	TPR_GAP: 0.83	FPR_GAP: 0.12	PPR_GAP: 0.13	
    2022-04-08 15:30:47 [INFO ]  Epoch:    7 [      0/  20018 ( 0%)]	Loss: -0.6750	 Data Time: 0.01s	Train Time: 1.77s
    2022-04-08 15:31:24 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:31:27 [INFO ]  Evaluation at Epoch 7
    2022-04-08 15:31:27 [INFO ]  Validation accuracy: 98.19	macro_fscore: 98.18	micro_fscore: 98.19	TPR_GAP: 0.96	FPR_GAP: 0.09	PPR_GAP: 0.14	
    2022-04-08 15:31:27 [INFO ]  Test accuracy: 98.59	macro_fscore: 98.58	micro_fscore: 98.59	TPR_GAP: 0.68	FPR_GAP: 0.11	PPR_GAP: 0.14	
    2022-04-08 15:31:28 [INFO ]  Epoch:    8 [      0/  20018 ( 0%)]	Loss: -0.6807	 Data Time: 0.01s	Train Time: 1.77s
    2022-04-08 15:32:04 [INFO ]  Epochs since last improvement: 2
    2022-04-08 15:32:07 [INFO ]  Evaluation at Epoch 8
    2022-04-08 15:32:07 [INFO ]  Validation accuracy: 98.28	macro_fscore: 98.27	micro_fscore: 98.28	TPR_GAP: 0.77	FPR_GAP: 0.07	PPR_GAP: 0.11	
    2022-04-08 15:32:07 [INFO ]  Test accuracy: 98.40	macro_fscore: 98.39	micro_fscore: 98.40	TPR_GAP: 0.82	FPR_GAP: 0.11	PPR_GAP: 0.13	
    2022-04-08 15:32:09 [INFO ]  Epoch:    9 [      0/  20018 ( 0%)]	Loss: -0.6773	 Data Time: 0.01s	Train Time: 1.75s
    2022-04-08 15:32:47 [INFO ]  Evaluation at Epoch 9
    2022-04-08 15:32:47 [INFO ]  Validation accuracy: 98.45	macro_fscore: 98.44	micro_fscore: 98.45	TPR_GAP: 0.81	FPR_GAP: 0.08	PPR_GAP: 0.11	
    2022-04-08 15:32:47 [INFO ]  Test accuracy: 98.80	macro_fscore: 98.79	micro_fscore: 98.80	TPR_GAP: 0.57	FPR_GAP: 0.10	PPR_GAP: 0.10	
    2022-04-08 15:32:49 [INFO ]  Epoch:   10 [      0/  20018 ( 0%)]	Loss: -0.6867	 Data Time: 0.01s	Train Time: 1.77s
    2022-04-08 15:33:25 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:33:28 [INFO ]  Evaluation at Epoch 10
    2022-04-08 15:33:28 [INFO ]  Validation accuracy: 98.50	macro_fscore: 98.48	micro_fscore: 98.50	TPR_GAP: 0.61	FPR_GAP: 0.06	PPR_GAP: 0.09	
    2022-04-08 15:33:28 [INFO ]  Test accuracy: 98.77	macro_fscore: 98.76	micro_fscore: 98.77	TPR_GAP: 0.65	FPR_GAP: 0.08	PPR_GAP: 0.09	
    2022-04-08 15:33:29 [INFO ]  Epoch:   11 [      0/  20018 ( 0%)]	Loss: -0.6892	 Data Time: 0.01s	Train Time: 1.76s
    2022-04-08 15:34:05 [INFO ]  Epochs since last improvement: 2
    2022-04-08 15:34:08 [INFO ]  Evaluation at Epoch 11
    2022-04-08 15:34:08 [INFO ]  Validation accuracy: 98.34	macro_fscore: 98.32	micro_fscore: 98.34	TPR_GAP: 0.79	FPR_GAP: 0.09	PPR_GAP: 0.11	
    2022-04-08 15:34:08 [INFO ]  Test accuracy: 98.67	macro_fscore: 98.66	micro_fscore: 98.67	TPR_GAP: 0.80	FPR_GAP: 0.09	PPR_GAP: 0.13	
    2022-04-08 15:34:10 [INFO ]  Epoch:   12 [      0/  20018 ( 0%)]	Loss: -0.6875	 Data Time: 0.01s	Train Time: 1.77s
    2022-04-08 15:34:49 [INFO ]  Evaluation at Epoch 12
    2022-04-08 15:34:49 [INFO ]  Validation accuracy: 98.70	macro_fscore: 98.69	micro_fscore: 98.70	TPR_GAP: 0.64	FPR_GAP: 0.07	PPR_GAP: 0.11	
    2022-04-08 15:34:49 [INFO ]  Test accuracy: 98.77	macro_fscore: 98.76	micro_fscore: 98.77	TPR_GAP: 0.80	FPR_GAP: 0.09	PPR_GAP: 0.14	
    2022-04-08 15:34:51 [INFO ]  Epoch:   13 [      0/  20018 ( 0%)]	Loss: -0.6881	 Data Time: 0.01s	Train Time: 1.83s
    2022-04-08 15:35:26 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:35:29 [INFO ]  Evaluation at Epoch 13
    2022-04-08 15:35:29 [INFO ]  Validation accuracy: 98.40	macro_fscore: 98.38	micro_fscore: 98.40	TPR_GAP: 0.69	FPR_GAP: 0.07	PPR_GAP: 0.09	
    2022-04-08 15:35:29 [INFO ]  Test accuracy: 98.64	macro_fscore: 98.63	micro_fscore: 98.64	TPR_GAP: 0.82	FPR_GAP: 0.09	PPR_GAP: 0.13	
    2022-04-08 15:35:31 [INFO ]  Epoch:   14 [      0/  20018 ( 0%)]	Loss: -0.6886	 Data Time: 0.01s	Train Time: 1.75s
    2022-04-08 15:36:06 [INFO ]  Epochs since last improvement: 2
    2022-04-08 15:36:09 [INFO ]  Evaluation at Epoch 14
    2022-04-08 15:36:09 [INFO ]  Validation accuracy: 98.49	macro_fscore: 98.48	micro_fscore: 98.49	TPR_GAP: 0.75	FPR_GAP: 0.10	PPR_GAP: 0.14	
    2022-04-08 15:36:09 [INFO ]  Test accuracy: 98.77	macro_fscore: 98.75	micro_fscore: 98.77	TPR_GAP: 0.79	FPR_GAP: 0.07	PPR_GAP: 0.10	
    2022-04-08 15:36:11 [INFO ]  Epoch:   15 [      0/  20018 ( 0%)]	Loss: -0.6904	 Data Time: 0.01s	Train Time: 1.74s
    2022-04-08 15:36:47 [INFO ]  Epochs since last improvement: 3
    2022-04-08 15:36:50 [INFO ]  Evaluation at Epoch 15
    2022-04-08 15:36:50 [INFO ]  Validation accuracy: 98.48	macro_fscore: 98.46	micro_fscore: 98.48	TPR_GAP: 0.59	FPR_GAP: 0.08	PPR_GAP: 0.12	
    2022-04-08 15:36:50 [INFO ]  Test accuracy: 98.80	macro_fscore: 98.79	micro_fscore: 98.80	TPR_GAP: 0.68	FPR_GAP: 0.07	PPR_GAP: 0.10	
    2022-04-08 15:36:52 [INFO ]  Epoch:   16 [      0/  20018 ( 0%)]	Loss: -0.6925	 Data Time: 0.01s	Train Time: 1.75s
    2022-04-08 15:37:27 [INFO ]  Epochs since last improvement: 4
    2022-04-08 15:37:30 [INFO ]  Evaluation at Epoch 16
    2022-04-08 15:37:30 [INFO ]  Validation accuracy: 98.48	macro_fscore: 98.47	micro_fscore: 98.48	TPR_GAP: 0.61	FPR_GAP: 0.06	PPR_GAP: 0.10	
    2022-04-08 15:37:30 [INFO ]  Test accuracy: 98.73	macro_fscore: 98.73	micro_fscore: 98.73	TPR_GAP: 0.73	FPR_GAP: 0.07	PPR_GAP: 0.10	
    2022-04-08 15:37:32 [INFO ]  Epoch:   17 [      0/  20018 ( 0%)]	Loss: -0.6918	 Data Time: 0.02s	Train Time: 1.74s
    2022-04-08 15:38:08 [INFO ]  Epochs since last improvement: 5
    2022-04-08 15:38:11 [INFO ]  Evaluation at Epoch 17
    2022-04-08 15:38:11 [INFO ]  Validation accuracy: 98.62	macro_fscore: 98.61	micro_fscore: 98.62	TPR_GAP: 0.65	FPR_GAP: 0.07	PPR_GAP: 0.09	
    2022-04-08 15:38:11 [INFO ]  Test accuracy: 98.85	macro_fscore: 98.84	micro_fscore: 98.85	TPR_GAP: 0.70	FPR_GAP: 0.08	PPR_GAP: 0.10	
    

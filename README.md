<!-- https://github.com/mhucka/readmine -->
# Fair_NLP_Classification

This is a PyTorch implementation collection of our recent work for bias mitigation in NLP classification.

Table of contents
-----------------

- [Fair_NLP_Classification](#fair_nlp_classification)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [Basic operation](#basic-operation)
    - [Additional options](#additional-options)
    - [Scripts for Large Scale Experiments](#scripts-for-large-scale-experiments)
    - [Add New Datasets/Models](#add-new-datasetsmodels)
  - [Model Selection and Fairness Evaluation](#model-selection-and-fairness-evaluation)
  - [Known issues and limitations](#known-issues-and-limitations)
  - [Getting help](#getting-help)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)


Introduction
------------

You can easily develop new algorithms, or readily apply existing algorithms.

The currently supported datasets include:
- Moji  
  Binary sentiment classification. Instances are annotated with AAE/SAE labels.
- Bios  
  Biography classification tasks with 28 profession classes.
  Instances are also annotated with gender and economy labels.  


The currently supported bias mitigation methods include:

- Learning Fair Representations
  - Adversarial Training  
  Li, Yitong, Timothy Baldwin and Trevor Cohn (2018) [Towards Robust and Privacy-preserving Text Representations](https://aclanthology.org/P18-2005/), ACL 2018.
  - Diverse Adversarial Training  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Diverse Adversaries for Mitigating Bias in Training](https://aclanthology.org/2021.eacl-main.239/), EACL 2021.
  - Decoupled Adversarial Training  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Decoupling Adversarial Training for Fair NLP](https://aclanthology.org/2021.findings-acl.41/), Findings of ACL 2021.
  - Towards Equal Opportunity Fairness through Adversarial Learning  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Towards Equal Opportunity Fairness through Adversarial Learning](), arXiv Pre-print.
  - Adversarial Regularization without Label  
  Working paper.
  - INLP  
    Ravfogel, Shauli, Yanai Elazar, Hila Gonen, Michael Twiton and Yoav Goldberg (2020) [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://aclanthology.org/2020.acl-main.647.pdf)
  - Contrastive Learning for Fair Representation
    Shen, Aili, Xudong Han, Trevor Cohn, Timothy Baldwin, and Lea Frermann (2021) [Contrastive Learning for Fair Representations](https://arxiv.org/abs/2109.10645), CoRR abs/2109.10645.

- Balanced Training
  - Instance Reweighting and Resampling  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Balancing out Bias: Achieving Fairness Through Training Reweighting](https://arxiv.org/abs/2109.08253), CoRR abs/2109.08253.
  - FairBatch: resampling dynamically during training  
  Roh, Yuji, Kangwook Lee, Steven Euijong Whang, and Changho Suh (2021) [FairBatch: Batch Selection for Model Fairness](https://arxiv.org/abs/2012.01696), ICLR, 2021

- Incorporating Demographic Factors  
  - Representation Augmentation  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Balancing out Bias: Achieving Fairness Through Training Reweighting](https://arxiv.org/abs/2109.08253), CoRR abs/2109.08253.

We are planning to add:
- \[algorithm\] 
- \[algorithm\] Minimizing Group Difference for Equal Opportunity Fairness

Prerequisites
------------

Start a new virtual environment:
```bash
conda create -n fair_nlp python=3.7 anaconda
conda activate fair_nlp

# additional python pacakges from requirements.txt
pip install -r requirements.txt
```

<!-- TODO Will we release the Bios dataset with economy labels? -->
Download the data used for this project:
```bash
./prepare_data.sh
```
 
Usage
-----

### Basic operation

```bash
python main.py --exp_id EXP_NAME
```

| Name                | Default value | Description                                                |
|---------------------|---------------|------------------------------------------------------------|
| project_dir         | dev           | protect step root: dev \| hypertune \| final               |
| exp_id              | test          | experiment id                                              |
| base_seed           | 1             | base random seed                                           |
| dataset             | Moji          | dataset: Moji \| Bios_gender \| Bios_economy \| Bios_both  |
| data_dir            | None          | dataset root, load from the default dir if set to None     |
| results_dir         | ./results/    | results directory                                          |
| model_dir           | models        | directory storing trained models                           |
| num_workers         | 0             | number of data loader workers                              |

The corresponding results such options and output logs will be saved in `results_dir/project_dir/dataset/exp_id`, while the model checkpoints and related predictions will be save in `results_dir/project_dir/dataset/exp_id/model_dir`.

- Train a vanilla model without debiasing

  - Train a mode over Moji with DeepMoji representations.
    ```bash
        python main.py --dataset Moji --emb_size 2304 --num_classes 2
    ```
  - Train a model over Bios with fixed BERT-base representations.
    ```bash
        python main.py --dataset Bios_gender --emb_size 768 --num_classes 28

        python main.py --dataset Bios_economy --emb_size 768 --num_classes 28

        python main.py --dataset Bios_both --emb_size 768 --num_classes 28
    ```

  - Train a model over Bios with trainable BERT-base representations.
    ```bash
        python main.py --dataset Bios_gender --emb_size 768 --num_classes 28 --encoder_architecture BERT --batch_size 16

        python main.py --dataset Bios_economy --emb_size 768 --num_classes 28 --encoder_architecture BERT --batch_size 16

        python main.py --dataset Bios_both --emb_size 768 --num_classes 28 --encoder_architecture BERT --batch_size 16
    ```

| Name                     | Default value | Description                                               |
|--------------------------|---------------|-----------------------------------------------------------|
| emb_size                 | 2304          | input embedding dimension                                 |
| num_classes              | 2             | number of target classes                                  |
| batch_size               | 1024          | input batch size for training                             |
| epochs                   | 100           | number of total epochs to train                           |
| lr                       | 0.003         | learning rate used to actually learn stuff                |
| epochs_since_improvement | 5             | terminate training for early stopping                     |
| device_id                | 0             | device id, -1 is cpu                                      |

- Train a model with adversarial debiasing

```bash
python main.py --adv_debiasing
```

| Name                 | Default value | Description                                            |
|----------------------|---------------|--------------------------------------------------------|
| adv_update_frequency | Batch         | Epoch \| Batch                                         |
| adv_level            | last_hidden   | input \| last_hidden \| output                         |
| adv_lambda           | 1             | strength of adversarial regularization                 |

- Ensemble adversarial training

```bash
python main.py --adv_debiasing --adv_num_subDiscriminator 3
```

| Name                     | Default value | Description                  |
|--------------------------|---------------|------------------------------|
| adv_num_subDiscriminator | 1             | number of sub-discriminators. |

- Diverse adversarial training

```bash
python main.py --adv_debiasing --adv_num_subDiscriminator 3 --adv_diverse_lambda 100
```

| Name                     | Default value | Description                                                                        |
|--------------------------|---------------|------------------------------------------------------------------------------------|
| adv_num_subDiscriminator | 1             | number of subdiscriminators.                                                       |
| adv_diverse_lambda       | 0             | strength of difference loss to encourage diverse representations for ensemble adv. |

- Towards conditional independence through adversarial training

```bash
python main.py --adv_debiasing --adv_gated
```

| Name              | Default value | Description                                                  |
|-------------------|---------------|--------------------------------------------------------------|
| adv_gated         | False         | gated discriminator for augmented inputs given target labels |
| adv_gated_mapping | One-hot       | mapping function from numerical labels to vectors.           |

- INLP

```bash
python main.py --INLP
```

| Name                            | Default value | Description                                                 |
|---------------------------------|---------------|-------------------------------------------------------------|
| INLP_discriminator_reweighting  | None          | if train the linear discriminator with reweighting          |
| INLP_by_class                   | False         | estimate the nullspace by_class                             |
| INLP_n                          | 300           | the maximum number of null-space projection iteration       |
| INLP_min_acc                    | 0.0           | ignore the iteration if the acc is lower than the threshold |


- Fair Contrastive Learning

```bash
python main.py --FCL
```

| Name                    | Default value | Description                                                     |
|-------------------------|---------------|-----------------------------------------------------------------|
| fcl_lambda_y            | 0.1           | strength of the supervised contrastive loss                     |
| fcl_lambda_g            | 0.1           | strength of the fair supervised contrastive loss                |
| fcl_temperature_y       | 0.01          | temperature for the fcl wrt main task learning                  |
| fcl_temperature_g       | 0.01          | temperature for the fcl wrt protected attribute unlearning      |
| fcl_base_temperature_y  | 0.01          | base temperature for the fcl wrt main task learning             |
| fcl_base_temperature_g  | 0.01          | base temperature for the fcl wrt protected attribute unlearning |

- Train a model with balanced training

```bash
python main.py --BT --BTObj joint
```

| Name       | Default value | Description                                                  |
|------------|---------------|--------------------------------------------------------------|
| BT         | False         | Reweighting or Resampling                                    |
| BTObj      | None          | joint \| y \| g \| stratified_y \| stratified_g \| EO        |
| full_label | True          | require full protected label                                 |

- Train a model with FairBach balanced training

```bash
python main.py --DyBT FairBach --DyBTObj stratified_y 
```

| Name       | Default value | Description                                                  |
|------------|---------------|--------------------------------------------------------------|
| DyBT       | False         | FairBach \| GroupDifference \| Others                        |
| DyBTObj    | None          | joint \| y \| g \| stratified_y \| stratified_g              |
| DyBTalpha  | 0.1           | a positive number for dynamic adjustment.                    |

- Train a model to incorporate demographic factors

```bash
python main.py --BT --BTObj joint --gated
```

| Name          | Default value | Description                                                  |
|---------------|---------------|--------------------------------------------------------------|
| gated         | False         | gated model for augmented inputs given demographic labels    |
| gated_mapping | One-hot       | mapping function from numerical labels to vectors.           |

### Additional options

- Logging

| Name                | Default value | Description                                                |
|---------------------|---------------|------------------------------------------------------------|
| base_seed           | 1             | base random seed                                           |
| log_interval        | 50            | how many batches to wait before logging training status    |
| checkpoint_interval | 1             | checkpoint interval (epoch)                                |
| no_log              | False         | if set, will not log into file                             |
| log_level           | INFO          | logging level, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL |

- Model architecture

| Name                | Default value | Description                                                              |
|---------------------|---------------|--------------------------------------------------------------------------|
| hidden_size         | 300           | number of hidden units of each hidden layer for the main task classifier |
| n_hidden            | 2             | number of hidden layers                                                  |
| dropout             | 0             | dropout probability                                                      |
| emb_size            | 2304          | input embedding dimension, i.e., dimension of sentence vectors           |
| num_classes         | 2             | number of target classes                                                 |
| activation_function | Tanh          | nonlinear activation function for the main task model                    |
| batch_norm          | False         | apply 1d batch norm to the model                                         |

- Discriminator architecture

| Name                    | Default value | Description                                                              |
|-------------------------|---------------|--------------------------------------------------------------------------|
| adv_hidden_size         | 300           | number of hidden units of each hidden layer for the main task classifier |
| adv_n_hidden            | 2             | number of hidden layers                                                  |
| adv_dropout             | 0             | dropout probability                                                      |
| adv_num_classes         | 2             | number of target classes                                                 |
| adv_activation_function | ReLu          | nonlinear activation function for the main task model                    |
| adv_batch_norm          | False         | apply 1d batch norm to the model                                         |

### Scripts for Large Scale Experiments

Performing experiments such as tuning hyperparameters requires generating lots of different scripts, we provide an example for generating required script automatically.

```bash
python scripts/gen_exps.py
```

### Add New Datasets/Models
Our implementation is friendly for adding new datasets and model architectures, please see docs for detailed instructions.

Model Selection and Fairness Evaluation
----------------------------
We provide implementation of DTO and DTO based model selection for different models. 

Please see `analysis\sample_model_selection.ipynb` for an example of loading training history, performing model selections based on different strategies, and creating basic plots.

Known issues and limitations
----------------------------

None are known at this time.


Getting help
------------

If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 

- Xudong Han (xudongh1@student.unimelb.edu.au)

or describe it in Issues.


Contributing
------------

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.


License
-------
This project is distributed under the terms of the [APACHE LICENSE, VERSION 2.0](https://www.apache.org/licenses/LICENSE-2.0).  The license applies to all files in the [GitHub repository](http://github.com/HanXudong/Fair_NLP_Classification) hosting this file.

Acknowledgments
---------------
* https://github.com/HanXudong/Decoupling_Adversarial_Training_for_Fair_NLP
* https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training
* https://github.com/SsnL/dataset-distillation
* https://github.com/huggingface/torchMoji
* https://github.com/mhucka/readmine
* https://github.com/yanaiela/demog-text-removal
* https://github.com/lrank/Robust_and_Privacy_preserving_Text_Representations
* https://github.com/yuji-roh/fairbatch
* https://github.com/shauli-ravfogel/nullspace_projection
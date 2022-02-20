<!-- https://github.com/mhucka/readmine -->
# Fair_NLP_Classification

This is a PyTorch implementation collection of our recent work for bias mitigation in NLP classification.

Table of contents
-----------------

- [Fair_NLP_Classification](#fair_nlp_classification)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Basic operation](#basic-operation)
    - [Additional options](#additional-options)
  - [Known issues and limitations](#known-issues-and-limitations)
  - [Getting help](#getting-help)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)


Introduction
------------

You can easily develop new algorithms, or readily apply existing algorithms.

<!-- The [Introduction](#introduction) section &ndash; which you are presently reading &ndash; should provide background for your software project, a brief explanation of what the project is about, and optionally, pointers to resources that can help orient readers.  (Some README recommendations go straight to installation instructions as the first section, but in my opinion, introductions are important for readers who are not familiar with your technical area.)  Ideally, this section should be short. -->

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
  - Deviser Adversarial Training  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Diverse Adversaries for Mitigating Bias in Training](https://aclanthology.org/2021.eacl-main.239/), EACL 2021.
  - Decoupled Adversarial Training
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Decoupling Adversarial Training for Fair NLP](https://aclanthology.org/2021.findings-acl.41/), Findings of ACL 2021.
  - Adversarial Regularization without Label  
  Working paper.
- Balanced Training
  - Instance Reweighting and Resampling  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Balancing out Bias: Achieving Fairness Through Training Reweighting](https://arxiv.org/abs/2109.08253), Arxiv.

- Incorporating Demographic Factors  
  - Representation Augmentation  
  Han, Xudong, Timothy Baldwin and Trevor Cohn (2021) [Balancing out Bias: Achieving Fairness Through Training Reweighting](https://arxiv.org/abs/2109.08253), Arxiv.

We are planning to add:
- \[dataset\] Hate Speech Detection
- \[algorithm\] Contrastive Learning for Fair Representation
- \[algorithm\] Minimizing Group Difference for Equal Opportunity Fairness

Installation
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
./download_data.sh
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
  - Train a model over Bios with BERT-base representations.
    ```bash
        python main.py --dataset Bios_gender --emb_size 768 --num_classes 28

        python main.py --dataset Bios_economy --emb_size 768 --num_classes 28

        python main.py --dataset Bios_both --emb_size 768 --num_classes 28
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

- Train a model with balanced training

```bash
python main.py --BT --BTObj joint
```

### Additional options

| Name                | Default value | Description                                                |
|---------------------|---------------|------------------------------------------------------------|
| base_seed           | 1             | base random seed                                           |
| log_interval        | 50            | how many batches to wait before logging training status    |
| checkpoint_interval | 1             | checkpoint interval (epoch)                                |
| no_log              | False         | if set, will not log into file                             |
| log_level           | INFO          | logging level, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL |


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


Acknowledgments
---------------
* https://github.com/HanXudong/Decoupling_Adversarial_Training_for_Fair_NLP
* https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training
* https://github.com/SsnL/dataset-distillation
* https://github.com/huggingface/torchMoji
* https://github.com/mhucka/readmine
* https://github.com/yanaiela/demog-text-removal
* https://github.com/lrank/Robust_and_Privacy_preserving_Text_Representations
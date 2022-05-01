# Fairlib

[Fairlib](https://github.com/HanXudong/fairlib) is a Python framework for assessing and improving classification fairness. Built-in algorithms can be applied to text inputs, structured inputs, and image inputs.  

The Fairlib package includes metrics for fairness evaluation, [algorithms for bias mitigation](https://github.com/HanXudong/fairlib/blob/main/docs/supported_bias_mitigation_algorithms.md), and functions for analysis.

For those who want to start with Fairlib now, you can try our [Colab Tutorial](https://colab.research.google.com/github/HanXudong/fairlib/blob/main/tutorial/demo.ipynb), which provides a gentle introduction to the concepts and capabilities. 
[The tutorials and other notebooks](https://github.com/HanXudong/fairlib/tree/main/tutorial) offer a deeper introduction. The complete API is also available.

## Table of contents

- [Fairlib](#fairlib)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Selection and Fairness Evaluation](#model-selection-and-fairness-evaluation)
  - [Known issues and limitations](#known-issues-and-limitations)
  - [Getting help](#getting-help)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)


## Installation

Fairlib currently requires Python3.7+ and [Pytorch](https://pytorch.org) 1.10 or higher.
Dependencies of the core modules are listed in [`requirements.txt`](https://github.com/HanXudong/fairlib/blob/main/requirements.txt). 
We *strongly* recommend using a [venv](https://docs.python.org/3/library/venv.html) or [conda](https://www.anaconda.com/) environment for installation.

**Standard Installation**

If you do not need further modifications, you can install it with:

```bash
# Start a new virtual environment:
conda create -n fairlib python=3.7
conda activate fairlib

pip install faircls
```

**Development Installation**

To set up a development environment, run the following commands to clone the repository and install
Fairlib:

```bash
git clone https://github.com/HanXudong/fairlib.git ~/fairlib
cd ~/fairlib; python setup.py develop
```

**Benchmark Datasets**  

Please refer to [data/README.md](https://github.com/HanXudong/fairlib/blob/main/data/README.md) for a list of fairness benchmark datasets.

## Usage

The full description of Fairlib usages can be found in [docs/usage](https://github.com/HanXudong/fairlib/blob/main/docs/usage.md). Here are the most basic examples.

- Fairlib can be run from the command line:
  ```bash
  python fairlib --exp_id EXP_NAME
  ```

- Fairlib can be imported as a package
  ```python
  from fairlib.base_options import options
  from src import networks

  config_file = 'opt.yaml'
  # Get options
  state = options.get_state(conf_file=config_file)

  # Init the model
  model = networks.get_main_model(state)

  # Training with debiasing
  model.train_self()
  ```

## Model Selection and Fairness Evaluation

We provide implementation of DTO and DTO based model selection for different models. 

Please see [this tutorial](https://github.com/HanXudong/fairlib/blob/main/tutorial/demo.ipynb) for an example of loading training history, performing model selections based on different strategies, and creating basic plots.
Moreover, [interactive plots](https://github.com/HanXudong/fairlib/blob/main/tutorial/interactive_plots.ipynb) are also supported, which can be used for analysis.

## Known issues and limitations

None are known at this time.


## Getting help

If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 

- Xudong Han (xudongh1@student.unimelb.edu.au)

or describe it in Issues.


## Contributing

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.


## License

This project is distributed under the terms of the [APACHE LICENSE, VERSION 2.0](https://www.apache.org/licenses/LICENSE-2.0).  The license applies to all files in the [GitHub repository](http://github.com/HanXudong/fairlib) hosting this file.

## Acknowledgments

* https://github.com/HanXudong/Decoupling_Adversarial_Training_for_Fair_NLP
* https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training
* https://github.com/SsnL/dataset-distillation
* https://github.com/huggingface/torchMoji
* https://github.com/mhucka/readmine
* https://github.com/yanaiela/demog-text-removal
* https://github.com/lrank/Robust_and_Privacy_preserving_Text_Representations
* https://github.com/yuji-roh/fairbatch
* https://github.com/shauli-ravfogel/nullspace_projection
* https://github.com/AiliAili/contrastive_learning_fair_representations
* https://github.com/AiliAili/Difference_Mean_Fair_Models
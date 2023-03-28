# *fairlib*


[*fairlib*](https://github.com/HanXudong/fairlib) is a Python framework for assessing and improving fairness. Built-in algorithms can be applied to text inputs, structured inputs, and image inputs.  

The *fairlib* package includes metrics for fairness evaluation, [algorithms for bias mitigation](https://hanxudong.github.io/fairlib/supported_bias_mitigation_algorithms.html), and functions for analysis.

For those who want to start with *fairlib* now, you can try our [Colab Tutorial](https://colab.research.google.com/github/HanXudong/fairlib/blob/main/tutorial/fairlib_demo.ipynb), which provides a gentle introduction to the concepts and capabilities. 
[The tutorials and other notebooks](https://hanxudong.github.io/fairlib/tutorial_interactive_demos.html) offer a deeper introduction. The [complete API](https://hanxudong.github.io/fairlib) is also available.

## Installation

*fairlib* currently requires Python3.8+ and [Pytorch](https://pytorch.org) 1.10 or higher.
Dependencies of the core modules are listed in [`requirements.txt`](https://github.com/HanXudong/fairlib/blob/main/requirements.txt). 
We *strongly* recommend using a [venv](https://docs.python.org/3/library/venv.html) or [conda](https://www.anaconda.com/) environment for installation.

**Standard Installation**

If you do not need further modifications, you can install it with:

```bash
# Start a new virtual environment:
conda create -n fairlib python=3.8
conda activate fairlib

pip install fairlib
```

**Development Installation**

To set up a development environment, run the following commands to clone the repository and install
*fairlib*:

```bash
git clone https://github.com/HanXudong/fairlib.git ~/fairlib
cd ~/fairlib
python setup.py develop
```

**Benchmark Datasets**  

Please refer to [data/README.md](https://github.com/HanXudong/fairlib/blob/main/data/README.md) for a list of fairness benchmark datasets.

## Usage

The full description of *fairlib* usages can be found in [*fairlib* cheat sheet](https://hanxudong.github.io/fairlib/tutorial_usage.html) and API reference. Here are the most basic examples.

- *fairlib* can be run from the command line:
  ```bash
  python fairlib --exp_id EXP_NAME
  ```

- *fairlib* can be imported as a package
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

Besides the classical loss- and performance-based model selection, we provide performance-fairness trade-off based model selection (see the paper below). 

Please see [this tutorial](https://hanxudong.github.io/fairlib/tutorial_notebooks/tutorial_Moji_demo.html) for an example of loading training history, performing model selections based on different strategies, and creating basic plots.
Moreover, [interactive plots](https://hanxudong.github.io/fairlib/tutorial_notebooks/tutorial_interactive_plots.html) are also supported, which can be used for analysis.

## Known issues and limitations

None are known at this time.


## Getting help

If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 

- Xudong Han (xudongh1@student.unimelb.edu.au)

or describe it in Issues.

## Paper

[fairlib: A Unified Framework for Assessing and Improving Classification Fairness](https://arxiv.org/abs/2205.01876)

Cite Us
```
@inproceedings{han-etal-2022-fairlib,
    title = "{F}air{L}ib: A Unified Framework for Assessing and Improving Fairness",
    author = "Han, Xudong  and
      Shen, Aili  and
      Li, Yitong  and
      Frermann, Lea  and
      Baldwin, Timothy  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-demos.7",
    pages = "60--71",
    abstract = "This paper presents FairLib, an open-source python library for assessing and improving model fairness. It provides a systematic framework for quickly accessing benchmark datasets, reproducing existing debiasing baseline models, developing new methods, evaluating models with different metrics, and visualizing their results.Its modularity and extensibility enable the framework to be used for diverse types of inputs, including natural language, images, and audio.We implement 14 debiasing methods, including pre-processing,at-training-time, and post-processing approaches. The built-in metrics cover the most commonly acknowledged fairness criteria and can be further generalized and customized for fairness evaluation.",
}

```


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

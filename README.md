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
  - [Authors and history](#authors-and-history)
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
  - Decoupled Adversarial Training
  - Deviser Adversarial Training
  - Adversarial Regularization with Demographic Label 

- Balanced Training
  - Instance Reweighting and Resampling    

- Incorporating Demographic Factors 
  - Representation Augmentation

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
```

<!-- TODO Will we release the Bios dataset with economy labels? -->
Download the data used for this project:
```bash
./download_data.sh
```
 
Usage
-----

This section explains the principles behind this README file.  If this repository were for actual _software_, this [Usage](#usage) section would explain more about how to run the software, what kind of output or behavior to expect, and so on.

### Basic operation

```bash
python main.py
```

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
| dataset                  | Moji          | dataset: Moji \| Bios_gender \| Bios_economy \| Bios_both |
| emb_size                 | 2304          | input embedding dimension                                 |
| num_classes              | 2             | number of target classes                                  |
| batch_size               | 1024          | input batch size for training                             |
| epochs                   | 100           | number of total epochs to train                           |
| lr                       | 0.003         | learning rate used to actually learn stuff                |
| epochs_since_improvement | 5             | terminate training for early stopping                     |
| device_id                | 0             | device id, -1 is cpu                                      |



### Additional options

Some projects need to communicate additional information to users and can benefit from additional sections in the README file.  It's difficult to give specific instructions &ndash; a lot depends on your software, your intended audience, etc.  Use your judgment and ask for feedback from users or colleagues to help figure out what else is worth explaining.


Known issues and limitations
----------------------------

none are known at this time


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

This README file is distributed under the terms of the [Creative Commons 1.0 Universal license (CC0)](https://creativecommons.org/publicdomain/zero/1.0/).  The license applies to this file and other files in the [GitHub repository](http://github.com/mhucka/readmine) hosting this file. This does _not_ mean that you, as a user of this README file in your software project, must also use CC0 license!  You may use any license for your work that you see fit.


Authors and history
---------------------------

In this section, list the authors and contributors to your software project.  (The original author of this file is [Mike Hucka](http://www.cds.caltech.edu/~mhucka/).)  Adding additional notes here about the history of the project can make it more interesting and compelling.


Acknowledgments
---------------

If your work was funded by any organization or institution, acknowledge their support here.  In addition, if your work relies on other software libraries, or was inspired by looking at other work, it is appropriate to acknowledge this intellectual debt too.  For example, in the process of developing this file, I used not only my own ideas and experiences &ndash; I read many (sometimes contradictory) recommendations for README files and examined real READMEs in actual use, and tried to distill the best ideas into the result you see here.  Sources included the following:

* http://tom.preston-werner.com/2010/08/23/readme-driven-development.html
* https://changelog.com/posts/top-ten-reasons-why-i-wont-use-your-open-source-project
* https://thoughtbot.com/blog/how-to-write-a-great-readme
* http://jonathanpeelle.net/making-a-readme-file
* https://github.com/noffle/art-of-readme
* https://github.com/noffle/common-readme
* https://github.com/RichardLitt/standard-readme
* https://github.com/jehna/readme-best-practices
* https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
* https://github.com/matiassingers/awesome-readme
* https://github.com/cfpb/open-source-project-template
* https://github.com/davidbgk/open-source-template/
* https://www.makeareadme.com
* https://github.com/lappleapple/feedmereadmes
* https://github.com/badges/shields
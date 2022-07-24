================
Bias Mitigation
================

.. list-table:: Built-in debiasing methods
   :header-rows: 1

   * - Type
     - Algorithm
     - Regression
     - Links
   * -  | Pre-
        |
        |
        |
     -  | BD
        | CB
        | JB
        | BTEO
     -  | Natural
        | Discretization
        | Discretization
        | Discretization
     -  | `Paper <https://aclanthology.org/D17-1323/>`__, `Github <https://github.com/uclanlp/reducingbias>`__, `Code <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/dataloaders/BT.py>`__
        | `Paper <https://arxiv.org/abs/1811.08489>`__, `Github <https://github.com/uvavision/Balanced-Datasets-Are-Not-Enough>`__, `Code <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/dataloaders/BT.py>`__
        | `Paper <https://arxiv.org/abs/2006.13114>`__, `Github <https://github.com/google-research/google-research/tree/master/group_agnostic_fairness>`__, `Code <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/dataloaders/BT.py>`__
        | `Paper <https://arxiv.org/abs/2109.08253>`__, `Github <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/dataloaders/generalized_BT.py>`__, `Code <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/dataloaders/BT.py>`__
   * -  | At-
        |
        |
        |
        |
        |
        |
        |
     -  | Adv
        | EAdv
        | DAdv
        | AADv & ADAdv
        | Gate
        | FairBatch
        | FairSCL
        | EO_CLA & EO_GLB
     -  | Natural
        | Natural
        | Natural
        | Discretization
        | Discretization
        | Discretization
        | Discretization
        | Discretization
     -  | `Paper <https://arxiv.org/abs/1805.06093>`__, `Github <https://github.com/lrank/Robust_and_Privacy_preserving_Text_Representations>`__, `Code <https://github.com/HanXudong/fairlib/tree/main/fairlib/src/networks/adv>`__
        | `Paper <https://arxiv.org/pdf/1808.06640.pdf>`__, `Github <https://github.com/yanaiela/demog-text-removal>`__, `Code <https://github.com/HanXudong/fairlib/tree/main/fairlib/src/networks/adv>`__
        | `Paper <https://arxiv.org/abs/2101.10001>`__, `Github <https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training>`__, `Code <https://github.com/HanXudong/fairlib/tree/main/fairlib/src/networks/adv>`__
        | `Paper <https://arxiv.org/pdf/2203.06317.pdf>`__, `Github <https://github.com/HanXudong/fairlib/tree/main/fairlib/src/networks/adv>`__, `Code <https://github.com/HanXudong/fairlib/tree/main/fairlib/src/networks/adv>`__
        | `Paper <https://arxiv.org/abs/2109.08253>`__, `Github <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/networks/augmentation_layer.py>`__, `Code <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/networks/augmentation_layer.py>`__
        | `Paper <https://arxiv.org/abs/2012.01696>`__, `Github <https://github.com/yuji-roh/fairbatch>`__, `Code <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/networks/DyBT/generalized_fairbatch.py>`__
        | `Paper <https://arxiv.org/abs/2109.10645>`__, `Github <https://github.com/AiliAili/contrastive_learning_fair_representations>`__, `Code <https://github.com/HanXudong/fairlib/tree/main/fairlib/src/networks/FairCL>`__
        | `Paper <https://arxiv.org/abs/2205.02393>`__, `Github <https://github.com/AiliAili/Difference_Mean_Fair_Models>`__, `Code <https://github.com/HanXudong/fairlib/blob/main/fairlib/src/networks/DyBT/gdl.py>`__
   * -  | Post-
        |
     -  | INLP
        | Soft Gated
     -  | Discretization
        | Discretization
     -  | `Paper <https://aclanthology.org/2020.acl-main.647/>`__, `Github <https://github.com/shauli-ravfogel/nullspace_projection>`__, `Code <https://github.com/HanXudong/fairlib/tree/main/fairlib/src/networks/INLP>`__
        | `Paper <https://arxiv.org/abs/2109.08253>`__, `Github <https://github.com/HanXudong/fairlib/blob/6dc60888748b2c04fc23eae101a56bcee078518a/fairlib/src/networks/classifier.py#L36-L39>`__, `Code <https://github.com/HanXudong/fairlib/blob/6dc60888748b2c04fc23eae101a56bcee078518a/fairlib/src/networks/classifier.py#L36-L39>`__

* A list of built-in debiasing methods, check out our `paper <http://arxiv.org/abs/2205.01876>`__ and `cheat sheet <https://hanxudong.github.io/fairlib/tutorial_usage.html>`__ for more details.
* All debiasing methods are designed for discrete outputs, but may not naturally support for continuous outputs. **Natural** refers to methods that naturally support continuous outputs, and **discretization** refers to discrete proxy label which is an built-in method to adopt classification 
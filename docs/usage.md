## Usage


### Basic operation

```bash
python fairlib --exp_id EXP_NAME
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
        python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --data_dir Path_to_data
    ```
  - Train a model over Bios with fixed BERT-base representations.
    ```bash
        python fairlib --dataset Bios_gender --emb_size 768 --num_classes 28 --num_groups 2 --data_dir Path_to_data

        python fairlib --dataset Bios_economy --emb_size 768 --num_classes 28 --num_groups 2 --data_dir Path_to_data

        python fairlib --dataset Bios_both --emb_size 768 --num_classes 28 --num_groups 4 --data_dir Path_to_data
    ```

  - Train a model over Bios with trainable BERT-base representations.
    ```bash
        python fairlib --dataset Bios_gender --emb_size 768 --num_classes 28 --encoder_architecture BERT --batch_size 16

        python fairlib --dataset Bios_economy --emb_size 768 --num_classes 28 --encoder_architecture BERT --batch_size 16

        python fairlib --dataset Bios_both --emb_size 768 --num_classes 28 --encoder_architecture BERT --batch_size 16
    ```

| Name                     | Default value | Description                                               |
|--------------------------|---------------|-----------------------------------------------------------|
| emb_size                 | 2304          | input embedding dimension                                 |
| num_classes              | 2             | number of target classes                                  |
| num_groups               | 2             | number of protected groups                                |
| batch_size               | 1024          | input batch size for training                             |
| epochs                   | 100           | number of total epochs to train                           |
| lr                       | 0.003         | learning rate used to actually learn stuff                |
| epochs_since_improvement | 5             | terminate training for early stopping                     |
| device_id                | 0             | device id, -1 is cpu                                      |

- Train a model with adversarial debiasing

```bash
python fairlib --adv_debiasing
```

| Name                 | Default value | Description                                            |
|----------------------|---------------|--------------------------------------------------------|
| adv_update_frequency | Batch         | Epoch \| Batch                                         |
| adv_level            | last_hidden   | input \| last_hidden \| output                         |
| adv_lambda           | 1             | strength of adversarial regularization                 |

- Ensemble adversarial training

```bash
python fairlib --adv_debiasing --adv_num_subDiscriminator 3
```

| Name                     | Default value | Description                  |
|--------------------------|---------------|------------------------------|
| adv_num_subDiscriminator | 1             | number of sub-discriminators. |

- Diverse adversarial training

```bash
python fairlib --adv_debiasing --adv_num_subDiscriminator 3 --adv_diverse_lambda 100
```

| Name                     | Default value | Description                                                                        |
|--------------------------|---------------|------------------------------------------------------------------------------------|
| adv_num_subDiscriminator | 1             | number of subdiscriminators.                                                       |
| adv_diverse_lambda       | 0             | strength of difference loss to encourage diverse representations for ensemble adv. |

- Towards conditional independence through adversarial training

```bash
python fairlib --adv_debiasing --adv_gated
```

| Name              | Default value | Description                                                  |
|-------------------|---------------|--------------------------------------------------------------|
| adv_gated         | False         | gated discriminator for augmented inputs given target labels |
| adv_gated_mapping | One-hot       | mapping function from numerical labels to vectors.           |

- INLP

```bash
python fairlib --INLP
```

| Name                            | Default value | Description                                                 |
|---------------------------------|---------------|-------------------------------------------------------------|
| INLP_discriminator_reweighting  | None          | if train the linear discriminator with reweighting          |
| INLP_by_class                   | False         | estimate the nullspace by_class                             |
| INLP_n                          | 300           | the maximum number of null-space projection iteration       |
| INLP_min_acc                    | 0.0           | ignore the iteration if the acc is lower than the threshold |


- Fair Contrastive Learning

```bash
python fairlib --FCL
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
python fairlib --BT Reweighting --BTObj joint
```

| Name       | Default value | Description                                                  |
|------------|---------------|--------------------------------------------------------------|
| BT         | False         | Reweighting or Resampling                                    |
| BTObj      | None          | joint \| y \| g \| stratified_y \| stratified_g \| EO        |
| full_label | True          | require full protected label                                 |

- Train a model with Dynamic balanced training

```bash
python fairlib --DyBT FairBatch --DyBTObj stratified_y 
```

| Name       | Default value | Description                                                  |
|------------|---------------|--------------------------------------------------------------|
| DyBT       | False         | FairBatch \| GroupDifference \| Others                       |
| DyBTObj    | None          | joint \| y \| g \| stratified_y \| stratified_g \| EO        |
| DyBTalpha  | 0.1           | a positive number for dynamic adjustment.                    |

For `Connecting Loss Difference with Equal Opportunity for Fair Models`, the **Diff** and **Mean** are: 
```bash
# L_diff as described in Section 3.2
python fairlib --DyBT GroupDifference --DyBTObj EO

# L_mean as described in Section 3.3 
python fairlib --DyBT GroupDifference --DyBTObj joint
```

- Train a model to incorporate demographic factors

```bash
python fairlib --BT --BTObj joint --gated
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
| adv_activation_function | ReLu          | nonlinear activation function for the main task model                    |
| adv_batch_norm          | False         | apply 1d batch norm to the model                                         |

### Add New Datasets/Models
Our implementation is friendly for adding new datasets and model architectures, please see docs for detailed instructions.

### Reproducibility
```bash
python fairlib --conf_file PATH_TO_EXP/opt.yaml
```

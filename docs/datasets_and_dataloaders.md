# Loading a Dataset

## Step 1. Implement the dataset class

A custom dataset class must implement the `load_data` function. Take a look at this sample implementation; the split is stored in a directory `self.data_dir`, which is determined by the `args.data_dir` and `split`. `args.data_dir` is either loaded from the arguments `--data_dir` or from the default value, which will be introduced into details later. While `split` has three possible string values, `{"train", "dev", "test"}`, indicating the split that will be loaded.

Then the `load_data` function must assign the value of `self.X` as inputs, `self.y` as target labels, and `self.protected_label` as the information for debiasing, such as gender, age, and race.

```python
from fairlib.dataloaders.utils import BaseDataset
from pathlib import Path

class SampleDataset(BaseDataset):

    def load_data(self):
        # Path to the data split.

        self.filename = "{}_df.pkl".format(self.split)
        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        # Assuming the size of the split is N
        self.X = data["X"] # N * input_size
        self.y = data["y"] # N, numerical labels, for example, {0,1}
        self.protected_label = data["protected_label"] # N, numerical labels

```

For more example, please take a look at `fairlib/src/dataloader/loaders.py`.

Debiasing methods such as instance reweighting will be automatically applied to the loaded dataset, which is implemented in `fairlib.src.dataloaders.utils.BaseDataset`.

## Step 2. Register the dataset

- `default_dataset_roots`

To avoid identifying the data root every time, we could predefine the default root the our dataset. we just need to add the path to the dictionary `default_dataset_roots` in `fairlib/src/dataloaders/__init__.py`. Take a look at the following example.
```python
    default_dataset_roots = dict(
        Moji='/data/deepmoji/split2/',
        # New dataset
        New_dataset='./data/path_to_dataset',
    )
```

- `get_dataloaders`
New, we need to register the dataset class for later use. Specially, all we need to do is adding an extra else if statement for the new dataset.

```python


def get_dataloaders(args):
    # Add the new dataset name here to avoid assert
    assert args.dataset in ["Moji", "Bios_gender", "Bios_economy", "Bios_both", "New_dataset"], "Not implemented"

    if args.dataset == "Moji":
        task_dataloader = DeepMojiDataset
    elif args.dataset in ["Bios_gender", "Bios_economy", "Bios_both"]:
        task_dataloader = BiosDataset
        args.protected_task = args.dataset.split("_")[1]
    # Add an additional else if statement for the new dataset
    elif args.dataset == "New_dataset":
        task_dataloader = SampleDataset # The SampleDataset class is the introduced in the previous section.
    else:
        pass
```

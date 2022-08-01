from fairlib.datasets.utils.download import download
from fairlib.src.utils import seed_everything
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..utils.preprocessing import onehot_encoder

target_variable = "income"
target_value = ">50K"

# Columns of interest
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

def convert_object_type_to_category(df):
    """Converts columns of type object to category."""
    df = pd.concat([
            df.select_dtypes(include=[], exclude=['object']),
            df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
            ], axis=1).reindex(df.columns, axis=1)
    return df

def preprocessing(tmp_df, mean_std_dict, vocab_dict):
    features = {}
    # Normalize numberiacal columns
    for col_name in mean_std_dict.keys():
        _mean, _std = mean_std_dict[col_name]
        features[col_name] = ((tmp_df[col_name]-_mean)/_std)
    # Encode categorical columns as indices
    for col_name in vocab_dict.keys():
        features[col_name] = tmp_df[col_name].map(
            {
                j:i for i,j in enumerate(vocab_dict[col_name])
            }
        )
    return pd.concat(features.values(), axis=1)

class Adult:

    _NAME = "Adult"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size

    def download_files(self):

        download(
                url= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                dest_folder = self.dest_folder
                )

        download(
                url= "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                dest_folder = self.dest_folder
                )

        data_file_path = os.path.join(self.dest_folder,  'adult.data')
        test_file_path = os.path.join(self.dest_folder,  'adult.test')
        
        # Drop the first line of test set, which is "|1x3 Cross validator"
        with open(test_file_path, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(test_file_path, 'w') as fout:
            fout.writelines([i[:-2]+"\n" for i in data[1:]])

        train_df = pd.read_csv(data_file_path,sep=',',names=columns)
        test_df = pd.read_csv(test_file_path,sep=',',names=columns)

        # Convert columns of type ``object`` to ``category`` 
        train_df = convert_object_type_to_category(train_df)
        test_df = convert_object_type_to_category(test_df)

        self.train_df = train_df
        self.test_df = test_df

    def processing(self):
        # Create splits
        test_df = self.test_df
        train_df = self.train_df

        cat_cols = train_df.select_dtypes(include='category').columns
        vocab_dict = {}
        for col in cat_cols:
            vocab_dict[col] = list(set(train_df[col].cat.categories)-{"?"})
        print(vocab_dict)

        temp_dict = train_df.describe().to_dict()
        mean_std_dict = {}
        for key, value in temp_dict.items():
            mean_std_dict[key] = [value['mean'],value['std']]
        print(mean_std_dict)

        train_df=preprocessing(train_df, mean_std_dict, vocab_dict)
        test_df=preprocessing(test_df, mean_std_dict, vocab_dict)

        encoder = onehot_encoder(["workclass","education","marital-status", "occupation", "relationship", "native-country"])
        encoder.fit(train_df)
        
        train_df = encoder.transform(train_df).dropna()
        test_df = encoder.transform(test_df).dropna()


        train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

        train_df.to_pickle(os.path.join(self.dest_folder, "Adult_train.pkl"))
        dev_df.to_pickle(os.path.join(self.dest_folder, "Adult_dev.pkl"))
        test_df.to_pickle(os.path.join(self.dest_folder, "Adult_test.pkl"))
        

    def prepare_data(self):
        self.download_files()
        self.processing()
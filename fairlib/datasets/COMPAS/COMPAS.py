from fairlib.datasets.utils.download import download
from fairlib.src.utils import seed_everything
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

dataset_file_name = 'compas-scores-two-years.csv'
target_variable = 'is_recid'
target_value = 'Yes'

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
    # One-hot encoding categorical features
    for col_name in ["c_charge_degree", "c_charge_desc", "age_cat"]:
        features[col_name] = pd.get_dummies(features[col_name], prefix=col_name)
    return pd.concat(features.values(), axis=1)

class COMPAS:

    _NAME = "COMPAS"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size

    def download_files(self):

        download(
                url= r"https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv", 
                dest_folder = self.dest_folder
                )

        file_path = os.path.join(self.dest_folder,dataset_file_name)
        temp_df = pd.read_csv(file_path)

        # Columns of interest
        columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                        'age', 
                        'c_charge_degree', 
                        'c_charge_desc',
                        'age_cat',
                        'sex', 'race',  'is_recid']

        # Drop duplicates
        temp_df = temp_df[['id']+columns].drop_duplicates()
        df = temp_df[columns].copy()

        # Convert columns of type ``object`` to ``category`` 
        df = pd.concat([
                df.select_dtypes(include=[], exclude=['object']),
                df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                ], axis=1).reindex(df.columns, axis=1)

        # Binarize target_variable
        df[target_variable] = df.apply(lambda x: target_value if x[target_variable]==1.0 else 'No', axis=1).astype('category')

        # Process protected-column values
        race_dict = {'African-American':'Black','Caucasian':'White'}
        df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 'Other', axis=1).astype('category')

        self.df = df

    def processing(self):
        # Create splits
        train_df, test_df = train_test_split(self.df, test_size=0.30, random_state=42)
        train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

        cat_cols = train_df.select_dtypes(include='category').columns
        vocab_dict = {}
        for col in cat_cols:
            vocab_dict[col] = list(set(train_df[col].cat.categories))

        temp_dict = train_df.describe().to_dict()
        mean_std_dict = {}
        for key, value in temp_dict.items():
            mean_std_dict[key] = [value['mean'],value['std']]

        train_df=preprocessing(train_df, mean_std_dict, vocab_dict)
        dev_df=preprocessing(dev_df, mean_std_dict, vocab_dict)
        test_df=preprocessing(test_df, mean_std_dict, vocab_dict)

        train_df.to_pickle(os.path.join(self.dest_folder, "COMPAS_train.pkl"))
        dev_df.to_pickle(os.path.join(self.dest_folder, "COMPAS_dev.pkl"))
        test_df.to_pickle(os.path.join(self.dest_folder, "COMPAS_test.pkl"))
        

    def prepare_data(self):
        self.download_files()
        self.processing()
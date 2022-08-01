from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class onehot_encoder:
    def __init__(self, categorical_columns):
        self.enc =  OneHotEncoder()
        self.categorical_columns = categorical_columns

    def fit(self,df):
        self.enc.fit(df[self.categorical_columns])

        column_keys = []
        for column_values, column_name in zip(self.enc.categories_, self.categorical_columns):
            for value in column_values:
                column_keys.append("_".join([column_name, str(value)]))
        self.column_keys = column_keys

    def transform(self, df, deop_orginal_colums = True):
        encoded_columns = self.enc.transform(df[self.categorical_columns]).toarray()
        encoded_df = pd.DataFrame(encoded_columns, columns=self.column_keys)
        encoded_df = df.join(encoded_df)
        if deop_orginal_colums:
            encoded_df.drop(self.categorical_columns, axis=1, inplace=True)
        return encoded_df
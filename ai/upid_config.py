import os
import pandas as pd


csv_file = './AIUPID_data.csv'

def generate_upid_table(csv_file):
    upid_data = pd.read_csv(csv_file)
    # upid_data.columns
    upid_data_1 = upid_data.set_index('label')

    # len(set(upid_data_1.index))
    #  find duplicated index...
    # upid_data_1.groupby('label').apply(
    #     lambda d: tuple(d.index) if len(d.index) > 1 else None
    # ).dropna()

    upid_data_2 = upid_data_1.drop(['Tree_Branch-pTwist'])
    upid_table = upid_data_2.to_dict(orient='index')
    return upid_table

UPID_Table = generate_upid_table(csv_file)

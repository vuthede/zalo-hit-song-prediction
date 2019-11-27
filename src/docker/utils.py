import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
from scipy.stats import sem
import numpy as np
import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import StratifiedKFold
import sys



def get_data(DATA_DIR):
    TRAININFO = os.path.join(DATA_DIR, "train_info.tsv")
    TRAINRANK = os.path.join(DATA_DIR, "train_rank.csv")
    TESTINFO = os.path.join(DATA_DIR, "test_info.tsv")

    # Prepare data
    df_i = pd.read_csv(TRAININFO, delimiter='\t', encoding='utf-8')
    df_r = pd.read_csv(TRAINRANK)
    df_i_train = df_i.merge(df_r, left_on='ID', right_on='ID')
    df_i_train["dataset"] = "train"

    df_i_test = pd.read_csv(TESTINFO, delimiter='\t', encoding='utf-8')
    df_i_test["label"] = np.nan
    df_i_test["dataset"] = "test"

    df = pd.concat([df_i_train, df_i_test])

    # Sort by ID
    df = df.sort_values(by=['ID'])
    df = df.reset_index()

    return df

def append_private_test_data(DATA_DIR):
    pass

def append_metadata(df, metadata_csv):
    assert os.path.isfile(metadata_csv), f"There is no file {metadata_csv}"

    df_track_info = pd.read_csv(metadata_csv)
    df = df.merge(df_track_info, left_on='ID', right_on='ID')

    return df


def print_rmse(df_train, oof):
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=99999)
    val_fold_index_lists = [val for (train, val) in folds.split(df_train.values, df_train.label.values)]
    folds_rmse = []
    for val_idx in val_fold_index_lists:
        numerator = (df_train.iloc[val_idx].label - oof[val_idx])**2
        inner = numerator/(len(df_train.iloc[val_idx]))
        fold_rmse = np.sqrt(inner.sum())
        folds_rmse.append(fold_rmse)
    mean_rmse = np.array(folds_rmse).mean()
    std_rmse = np.array(folds_rmse).std()
    print(f'Mean val RMSE {mean_rmse:.4f} +/- {std_rmse:.4f}' )
    return mean_rmse, std_rmse




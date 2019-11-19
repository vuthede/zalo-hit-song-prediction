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


def plot_val_curve(train_scores, test_scores, param_range):
    """
    Often a “one-standard
    error” rule is used with cross-validation, in which we choose the most par-
    simonious model whose error is no more than one standard error above
    the error of the best model. The Elements of
    Statistical Learning, Data Mining, Inference, and Prediction by Trevor Hastie
    """
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    train_scores_stderr = sem(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    test_scores_stderr = sem(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score (SEM shown)")
    #plt.ylim(-0.1, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw, marker='o')
    plt.fill_between(param_range, train_scores_mean - train_scores_stderr,
                     train_scores_mean + train_scores_stderr, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw, marker='o')
    plt.fill_between(param_range, test_scores_mean - test_scores_stderr,
                     test_scores_mean + test_scores_stderr, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def print_data_types(df, column_names):
    for i in [(col, df[col].dtype, df[col].nunique()) for col in df[column_names].columns]:
        print(i)
        print()

def get_data(DATA_DIR):
    TRAININFO = os.path.join(DATA_DIR, "train_info.tsv")
    TRAINRANK = os.path.join(DATA_DIR, "train_rank.csv")
    TESTINFO = os.path.join(DATA_DIR, "test_info.tsv")
    SUBMISSION = os.path.join(DATA_DIR, "submission.csv")

    # Prepare data
    df_i = pd.read_csv(TRAININFO, delimiter='\t', encoding='utf-8')
    df_r = pd.read_csv(TRAINRANK)
    df_i_train = df_i.merge(df_r, left_on='ID', right_on='ID')
    df_i_train["dataset"] = "train"

    df_i_test = pd.read_csv(TESTINFO, delimiter='\t', encoding='utf-8')
    df_i_test["label"] = np.nan
    df_i_test["dataset"] = "test"

    df = pd.concat([df_i_train, df_i_test])
    df_track_info = pd.read_csv(os.path.join(DATA_DIR, "all_track_info.csv"))
    df = df.merge(df_track_info, left_on='ID', right_on='ID')
    df_audio_features = pd.read_csv(os.path.join(DATA_DIR, "all_track_audio_features.csv"))
    df = df.merge(df_audio_features, left_on="ID", right_on="ID", how="left")
    df_album_hash = pd.read_csv(os.path.join(DATA_DIR, "album_hash.csv"))
    df = df.merge(df_album_hash, left_on="ID", right_on="ID", how="left")

    # Sort by ID
    df = df.sort_values(by=['ID'])
    df = df.reset_index()

    df.head(2)

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

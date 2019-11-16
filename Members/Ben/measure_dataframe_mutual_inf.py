from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif

def get_numerical_mutual_info(df):
    df_train = df[df.dataset=="train"]
    n_splits=3
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99999)

    labels= df_train.label
    numeric_data = df_train[all_features_in_order_list].select_dtypes(include=['float64', 'int64'])
    numeric_information_series = pd.Series( {name:0 for name in numeric_data.columns})
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, df_train.label.values)):
        print("cross validation ", fold_)
        train_df_fold = numeric_data.iloc[trn_idx]
        label_fold = df_train.label.iloc[trn_idx]
        train_df_fold.fillna(train_df_fold.mean(), inplace=True)  # data leakage unless done inside folds
        numeric_information = {}
        numeric_information = {name:score for (score, name) in zip(mutual_info_regression(train_df_fold, label_fold), numeric_data.columns)}
        numeric_information_series = numeric_information_series + pd.Series(numeric_information)
    return (pd.Series(numeric_information_series)/n_splits).sort_values()

def get_categorical_mutual_info():
    # TODO rename variables from numeric to categorical
    df_train = df[df.dataset=="train"]
    n_splits=3
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99999)

    labels= df_train.label
    numeric_data = df_train[all_features_in_order_list].select_dtypes(include=['category'])
    numeric_information_series = pd.Series( {name:0 for name in numeric_data.columns})
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, df_train.label.values)):
        print("cross validation ", fold_)
        train_df_fold = numeric_data.iloc[trn_idx]
        label_fold = df_train.label.iloc[trn_idx]
        train_df_fold.fillna(train_df_fold.mean(), inplace=True)  # data leakage unless done inside folds
        numeric_information = {}
        numeric_information = {name:score for (score, name) in zip(mutual_info_regression(train_df_fold, label_fold), numeric_data.columns)}
        numeric_information_series = numeric_information_series + pd.Series(numeric_information)
    return (pd.Series(numeric_information_series)/n_splits).sort_values()

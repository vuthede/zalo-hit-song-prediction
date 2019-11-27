import pandas as pd
import numpy as np
import random
import os
import lightgbm as lgb
import sys
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.externals import joblib
import pandas as pd
from sklearn.externals import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from sklearn.utils import compute_sample_weight
from format_features import create_album_score_lookup_table, create_artist_score_lookup_table, assign_value_redesigned
from format_features import baysianEncodeFeature
from format_features import format_features, assign_artist_features_inplace
from typecast_features import cast_cat_dtype_to_cat_codes
from typecast_features import typecast_features
from utils import print_data_types
from utils import get_data, print_rmse


np.random.seed(1)
random.seed(1)
DATA_DIR="/media/DATA/zalo-hit-song-prediction/csv/"
df = get_data(DATA_DIR)
df = format_features(df)
all_features_in_order_list, df = typecast_features(df, cast_to_catcode=True)

print("Len before: ",len(df) )
# Remove len =0
df = df[(df.length>0) | (df.num_same_title==1)]

print("Len after: ",len(df) )
df = assign_artist_features_inplace(df)

###

chosen_features = ["album_right", "istrack11", "no_artist", "no_composer", "freq_artist", "freq_composer", "year",
                   "month", "hour", "day", "len_of_songname",
                   "isRemix", "isOST", "isBeat", "isVersion", "isCover", "num_song_release_in_final_month",
                   "length", "genre", "track", "album_artist", "islyric", "album_artist_contain_artistname",
                   "len_album_name", "isRemixAlbum", "isOSTAlbum", "isSingleAlbum", "album_name_is_title_name",
                   "isBeatAlbum", "isCoverAlbum", "artist_name", "composers_name", "copyright",
                   "artist_id_min_cat", "composers_id_min_cat", "artist_id_max_cat", "composers_id_max_cat",
                   "freq_artist_min", "freq_composer_min", "dayofyear", "weekday", "isHoliday",
                   "num_album_per_min_artist", "num_album_per_min_composer",
                   "numsongInAlbum", "isSingleAlbum_onesong", "artist_mean_id",
                   "artist_std_id", "artist_count_id", "title_cat", "num_same_title"]

chosen_features += ["predicted_label"]
# chosen_features += ["mean_album_score", "mean_artist_min_score"]
df_train = df[df.dataset == "train"]
df_test = df[df.dataset == "test"]
num_boost_round = 1000000
early_stopping_rounds = 20000
# LGBM configuration
alg_conf = {
    "num_boost_round":num_boost_round,
    "max_depth" : -1,
    "num_leaves" : 50,
    'learning_rate' : 0.001,
    'boosting_type' : 'gbdt',
    'objective' : 'root_mean_squared_error',
    'metric':'root_mean_squared_error',
    "early_stopping_rounds": early_stopping_rounds, #
    'bagging_freq': 25,
    'bagging_fraction':0.95,
    'boost_from_average':'false',
    'feature_fraction':0.1,
    'min_data_in_leaf':5,
    'num_threads':8,
    'tree_learner':'serial',
    'objective':'regression',
    'reg_alpha': 0.1002650970728192,
    'reg_lambda':0.1003427518866501,
    'verbosity':1,
    'seed':99999,
    'use_missing':True,
    # adding default arguments
    'subsample_for_bin' : 200000,
    'min_split_gain':0,
    'min_child_weight':0.001,
    'min_child_samples':20,
    'subsample':1,
    'colsample_bytree':1,
    'importance_type':'split',
}

# Calling Regressor using scikit-learn API 
sk_reg = lgbm.sklearn.LGBMRegressor(
    num_leaves=alg_conf["num_leaves"], 
    n_estimators=num_boost_round, 
    max_depth=alg_conf["max_depth"],
    learning_rate=alg_conf["learning_rate"],
    objective=alg_conf["objective"],
    metric=alg_conf['metric'],
    #min_sum_hessian_in_leaf=alg_conf["min_child_weight"],
    #min_data_in_leaf=alg_conf["min_child_samples"],
    min_data_in_leaf=alg_conf['min_data_in_leaf'],
    min_child_weight=alg_conf['min_child_weight'],
    bagging_freq=alg_conf['bagging_freq'],
    bagging_fraction=alg_conf['bagging_fraction'],
    boost_from_average=alg_conf['boost_from_average'],
    feature_fraction=alg_conf['feature_fraction'],

    num_threads=alg_conf['num_threads'],
    tree_learner=alg_conf['tree_learner'],
    reg_alpha=alg_conf['reg_alpha'],
    reg_lambda=alg_conf['reg_lambda'],
    verbosity=alg_conf['verbosity'],
    seed=alg_conf['seed'],
    #adding defaults
    subsample_for_bin=alg_conf['subsample_for_bin'],
    min_split_gain=alg_conf['min_split_gain'],

    min_child_samples=alg_conf['min_child_samples'],
    subsample=alg_conf['subsample'],
    colsample_bytree=alg_conf['colsample_bytree'],
    importance_type=alg_conf['importance_type']
)

from math import sqrt

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=99999)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
labels = df_train.label
best_stopping_iterations_list = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, df_train.album_right.values)):
    print("Fold {}".format(fold_))
    # Create lookup table
    album_lookup_table = create_album_score_lookup_table(df_train.iloc[trn_idx])
    artist_lookup_table = create_artist_score_lookup_table(df_train.iloc[trn_idx])
    df_train["predicted_label"] = [assign_value_redesigned(album_lookup_table, artist_lookup_table, r) for i, r in
                                   df_train.iterrows()]
    df_test["predicted_label"] = [assign_value_redesigned(album_lookup_table, artist_lookup_table, r) for i, r in
                                  df_test.iterrows()]
    print("No imputation: Train\n")
    print(df_train.iloc[trn_idx][chosen_features].isnull().sum())
    print("No imputation: Val\n")
    print(df_train.iloc[val_idx][chosen_features].isnull().sum())
    print("No imputation: Test\n")
    print(df_test[chosen_features].isnull().sum())

    train_weights = compute_sample_weight('balanced', df_train.iloc[trn_idx].label)
    
    #trn_data = lgb.Dataset(df_train.iloc[trn_idx][chosen_features],
    #                       label=labels.iloc[trn_idx],params={'verbose': -1},
    #                       free_raw_data=False,
    #                       weight=train_weights)

    #val_data = lgb.Dataset(df_train.iloc[val_idx][chosen_features], label=labels.iloc[val_idx],params={'verbose': -1}, free_raw_data=False)
    clf = sk_reg.fit(df_train.iloc[trn_idx][chosen_features], labels.iloc[trn_idx],
                     eval_set=(df_train.iloc[val_idx][chosen_features], labels.iloc[val_idx]),
                     early_stopping_rounds=early_stopping_rounds,
                     sample_weight=train_weights,
                     eval_metric='root_mean_squared_error',
                     verbose=5000)


    #clf = lgb.train(param, trn_data, 1000000, valid_sets=[trn_data, val_data], verbose_eval=5000,
    #                early_stopping_rounds=20000)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][chosen_features], num_iteration=clf.best_iteration)
    predictions += clf.predict(df_test[chosen_features], num_iteration=clf.best_iteration) / folds.n_splits
    best_stopping_iterations_list.append(clf.best_iteration)

print("RMSE: {:<8.5f}".format(sqrt(mean_squared_error(df_train.label, oof))))
sub = pd.DataFrame({"ID": df_test.ID.values})
sub["label"] = predictions.round(decimals=4)
mean_rmse, std_rmse = print_rmse(df_train, oof)
sub.to_csv(f"baseline5_sklearn_{mean_rmse:.4f}_{std_rmse:.4f}.csv", index=False, header=False)
print("The number of best number of iterations was:", np.array(best_stopping_iterations_list).mean(), "+/-", np.array(best_stopping_iterations_list).std())
 

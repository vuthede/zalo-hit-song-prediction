short_categoricals = ['genre',
 'album_name_is_title_name',
 'year',
 'copyright',
 'artist_name',
 'isHoliday',
 'album_artist_contain_artistname',
 'day',
 'numsongInAlbum',
 'month',
 'weekday',
 'hour']



import sys
sys.path.insert(0, '/media/DATA/zalo-hit-song-prediction/src/')

import pandas as pd
import numpy as np
import random
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.externals import joblib
import pandas as pd
from format_features import baysianEncodeFeature
from utils import print_data_types
from utils import get_data
from format_features import format_features, assign_artist_features_inplace
from typecast_features import typecast_features

np.random.seed(1)
random.seed(1)
DATA_DIR = "/media/DATA/zalo-hit-song-prediction/csv/"
df = get_data(DATA_DIR)
df = format_features(df)
all_features_in_order_list, df = typecast_features(df, cast_to_catcode=True)
df = assign_artist_features_inplace(df)

###
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from format_features import create_album_score_lookup_table, create_artist_score_lookup_table, assign_value
from sklearn.model_selection import train_test_split
from typecast_features import cast_cat_dtype_to_cat_codes

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
                   "artist_std_id", "artist_count_id", "title_truncated", "num_same_title"]

chosen_features += ["predicted_label"]
# chosen_features += ["mean_album_score", "mean_artist_min_score"]
df_train = df[df.dataset == "train"]
df_test = df[df.dataset == "test"]

param = {
    'bagging_freq': 20,
    'bagging_fraction': 0.95, 'boost_from_average': 'false',
    'boost': 'gbdt', 'feature_fraction': 0.1, 'learning_rate': 0.001,
    'max_depth': -1, 'metric': 'root_mean_squared_error', 'min_data_in_leaf': 5,
    'num_leaves': 50,
    'num_threads': 8, 'tree_learner': 'serial', 'objective': 'regression',
    'reg_alpha': 0.1002650970728192, 'reg_lambda': 0.1003427518866501, 'verbosity': 1,
    "seed": 99999,
    "use_missing": True
}

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=99999)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
labels = df_train.label
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, df_train.label.values)):
    print("Fold {}".format(fold_))

    # Create lookup table
    album_lookup_table = create_album_score_lookup_table(df_train.iloc[trn_idx])
    artist_lookup_table = create_artist_score_lookup_table(df_train.iloc[trn_idx])

    df_train["predicted_label"] = [assign_value(album_lookup_table, artist_lookup_table, r) for i, r in
                                   df_train.iterrows()]
    print("Percentage null in valid:", len(np.sum(df_train.iloc[val_idx].isnull())) / len(df_train.iloc[val_idx]))
    df_test["predicted_label"] = [assign_value(album_lookup_table, artist_lookup_table, r) for i, r in
                                  df_test.iterrows()]
    print("Percentage null in test:", len(np.sum(df_test.isnull())) / len(df_test))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][chosen_features], label=labels.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][chosen_features], label=labels.iloc[val_idx])
    clf = lgb.train(param, trn_data, 1000000, valid_sets=[trn_data, val_data], verbose_eval=5000,
                    early_stopping_rounds=20000)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][chosen_features], num_iteration=clf.best_iteration)
    predictions += clf.predict(df_test[chosen_features], num_iteration=clf.best_iteration) / folds.n_splits

from math import sqrt

print("RMSE: {:<8.5f}".format(sqrt(mean_squared_error(df_train.label, oof))))
sub = pd.DataFrame({"ID": df_test.ID.values})
sub["label"] = predictions.round(decimals=4)
mean_rmse, std_rmse = print_rmse(df_train, oof)
sub.to_csv(f"baseline2_album_right_titletruncated_{mean_rmse:.4f}_{std_rmse:.4f}_rep.csv", index=False, header=False)


# coding: utf-8

# In[4]:


import pandas as pd
import os
import random
import numpy as np
np.random.seed(99999)
random.seed(99999)

np.random.seed(1)
random.seed(1)
DATA_DIR = "C:\\Users\\Ben\\OneDrive\\ZaloSongComp\\zalo-hit-song-prediction\\csv"
TRAININFO = os.path.join(DATA_DIR, "train_info.tsv")
TRAINRANK =  os.path.join(DATA_DIR, "train_rank.csv")
TESTINFO = os.path.join(DATA_DIR, "test_info.tsv")
SUBMISSION = os.path.join(DATA_DIR, "submission.csv")

# Prepare data
df_i = pd.read_csv(TRAININFO, delimiter='\t',encoding='utf-8')
df_r = pd.read_csv(TRAINRANK)
df_i_train = df_i.merge(df_r, left_on='ID', right_on='ID')
df_i_train["dataset"] = "train"

df_i_test = pd.read_csv(TESTINFO, delimiter='\t',encoding='utf-8')
df_i_test["label"] = np.nan
df_i_test["dataset"] = "test"

df = pd.concat([df_i_train, df_i_test])
df_track_info = pd.read_csv("../csv/all_track_info.csv")
df = df.merge(df_track_info, left_on='ID', right_on='ID')
df_audio_features = pd.read_csv("../csv/all_track_audio_features.csv")
df =df.merge(df_audio_features,left_on="ID",right_on="ID", how="left")

# Sort by ID
df = df.sort_values(by=['ID'])
df= df.reset_index()

df.head(5)


# In[6]:


from format_features import format_features
df = format_features(df)


# In[8]:


df_train = df[df.dataset == "train"]
df_test = df[df.dataset == "test"]


# In[9]:


feature = "isRemix"
print(df_train[feature].describe(include="all"))
df_train[feature].head()


# In[14]:


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt

import dtreeviz
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
get_ipython().run_line_magic('matplotlib', 'notebook')

chosen_features_dict = {"no_artist":"int64",
                        "no_composer":"int64",
                        "freq_artist":"int64", 
                        "freq_composer":"int64",
                        "year":"category",
                        "month":"category", 
                        "day":"category",
                        "len_of_songname":"int64",
                        "isRemix":"category",
                        "isOST":"category",
                        "isBeat":"category",
                        "isVersion":"category",
                        "isCover":"category",
                        "num_song_release_in_final_month":"int64",                   
                        "length":"float64",
                        "genre":"category", 
                        "track":"float64", # float between 0 and 1 representing track_num/total_tracks
                        "album_artist":"category", # album artist name
                        "album":"category", # album name
                        "islyric":"category",
                        "album_artist_contain_artistname":"category",
                        "len_album_name":"int64", # ordinal 
                        "isRemixAlbum":"category",
                        "isOSTAlbum":"category",
                        "isSingleAlbum":"category", 
                        "album_name_is_title_name":"category",
                        "isBeatAlbum":"category",
                        "isCoverAlbum":"category",
                        "artist_name_cat":"category",
                        "composers_name_cat":"category",
                        "copyright_cat":"category" ,
                        "artist_id_min_cat":"category", 
                        "composers_id_min_cat":"category", 
                        "artist_id_max_cat":"category", 
                        "composers_id_max_cat":"category"}
chosen_features = list(chosen_features_dict.keys())
for feat_name, feat_type in chosen_features_dict.items():
    df[feat_name] = df[feat_name].astype(feat_type)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99999)
X_train, X_valid, y_train, y_valid = train_test_split(df_train[chosen_features], df_train.label, test_size=0.2, random_state=99999)

clf = tree.DecisionTreeRegressor(max_depth=5) # DecisionTreeClassifier
#lgb.cv(param, train_data, num_round, nfold=5)

result = clf.fit(X_train, y_train)
fig, ax = plt.subplots()
f = tree.plot_tree(result, feature_names=chosen_features, proportion=True) 
X_train.shape


viz = dtreeviz(clf, X_train, y_train, 
               feature_names=chosen_features,
               target_name="rank"
               )
viz.save("boston.svg") # suffix determines the generated image format
viz.view()     


# In[12]:


from math import sqrt
print("RMSE: {:<8.5f}".format(sqrt(mean_squared_error(y_valid, clf.predict(X_valid)))))


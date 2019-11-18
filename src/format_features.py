import numpy as np
import seaborn as sns
import re
from dateutil import relativedelta
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval as make_tuple
import pandas as pd
import seaborn as sns
from datetime import date
import holidays
from dateutil import relativedelta
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval as make_tuple
from scipy.stats import ttest_ind
from functools import reduce
from datetime import datetime
import re

def format_features(df):
    # Fill missing copyright across train and test with fixed values
    df["copyright"] = df["copyright"].fillna("UnavailableInformation")

    # Fill nan album
    print("There is {} ratio is nan album".format(len(df[df["album"].isnull()]) / len(df)))
    df["album_raw_from_mp3_metadata"] = df["album"]
    df["album"] = df["album"].fillna("")
    df["len_album_name"] = df["album"].apply(lambda x: len(x.split(" ")))
    df["isRemixAlbum"] = [1 if "Remix" in t else 0 for t in df["album"]]
    df["isOSTAlbum"] = [1 if "OST" in t else 0 for t in df["album"]]
    df["isSingleAlbum"] = [1 if "Single" in t else 0 for t in df["album"]]
    df["isBeatAlbum"] = [1 if "Beat" in t else 0 for t in df["album"]]
    df["isTopHitAlbum"] = [1 if "Top Hits" in t else 0 for t in df["album"]]
    df["isCoverAlbum"] = [1 if "Cover" in t else 0 for t in df["album"]]
    df["isEPAlbum"] = [1 if "EP" in t else 0 for t in df["album"]]
    df["isLienKhucAlbum"] = [1 if "Liên Khúc" in t else 0 for t in df["album"]]

    df["album_name_is_title_name"] = [1 if r.title in r.album else 0 for i, r in df.iterrows()]

    # Fill genre
    print("There is {} ratio is nan genre".format(len(df[df["genre"].isnull()]) / len(df)))
    df["genre"] = df["genre"].fillna("No genre")
    # Fill album_artist

    print("There is {} ratio is nan album_artist".format(len(df[df["album_artist"].isnull()]) / len(df)))
    df["album_artist"] = df["album_artist"].fillna("No album_artist")
    df["album_artist_contain_artistname"] = [1 if r.album_artist in r.artist_name else 0 for i, r in df.iterrows()]

    # Fill track
    print("There is {} ratio is nan track".format(len(df[df["track"].isnull()]) / len(df)))
    df["track"] = df["track"].fillna("(1, 1)")
    df["istrack11"] = df["track"] == "(1, 1)"

    def tracknum_to_value(track_num):
        try:

            track_num = make_tuple(track_num)
            if track_num[0] is not None:
                return float(track_num[0]) / float(track_num[1])
            else:
                return 1.0
        except:
            return 1.0

    df["track"] = df["track"].apply(lambda t: tracknum_to_value(t))

    # Fill lyric
    print("There is {} ratio is nan lyric".format(len(df[df["lyric"].isnull()]) / len(df)))
    df["lyric"] = df["lyric"].fillna("")
    df["islyric"] = df["lyric"].apply(lambda x: True if len(x) else False)
    df["num_line_lyric"] = df["lyric"].apply(lambda x: len(x.split("\r")))

    # --------------------------------------------------------

    df['no_artist'] = df.artist_name.apply(lambda x: len(x.split(",")))
    df['no_composer'] = df.composers_name.apply(lambda x: len(x.split(",")))
    
    df["datetime"] = pd.to_datetime(df.release_time)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["weekday"] = df["datetime"].dt.weekday

    in_holidays = holidays.HolidayBase()
    for i in range(26, 32):
        in_holidays.append(str(i) + '-01-2017')
    in_holidays.append('01-02-2017')
    for i in range(14, 21):
        in_holidays.append(str(i) + '-02-2018')
    in_holidays.append('30-04-2017')
    in_holidays.append('30-04-2018')
    in_holidays.append('01-01-2017')
    in_holidays.append('01-01-2018')
    in_holidays.append('14-02-2017')
    in_holidays.append('14-02-2018')
    in_holidays.append('08-03-2017')
    in_holidays.append('08-03-2018')
    in_holidays.append('01-05-2017')
    in_holidays.append('01-05-2018')
    in_holidays.append('06-04-2017')
    in_holidays.append('25-04-2018')
    in_holidays.append('01-06-2017')
    in_holidays.append('01-06-2018')
    in_holidays.append('04-10-2017')
    in_holidays.append('24-09-2018')
    in_holidays.append('20-10-2017')
    in_holidays.append('20-10-2018')
    in_holidays.append('20-11-2017')
    in_holidays.append('20-11-2018')
    in_holidays.append('24-12-2017')
    in_holidays.append('24-12-2018')
    df['isHoliday'] = df.release_time.apply(lambda x: x in in_holidays)

    df["len_of_songname"] = df["title"].apply(lambda x: len(x.split(" ")))
    df["isRemix"] = [1 if "Remix" in t else 0 for t in df["title"]]
    df["isOST"] = [1 if "OST" in t else 0 for t in df["title"]]
    df["isBeat"] = [1 if "Beat" in t else 0 for t in df["title"]]
    df["isVersion"] = [1 if "Version" in t else 0 for t in df["title"]]
    df["isCover"] = [1 if "Cover" in t else 0 for t in df["title"]]
    df["isLienKhuc"] = [1 if "Liên Khúc" in t else 0 for t in df["title"]]

    df["day_release"] = df.groupby(["year", "dayofyear"]).ngroup().astype("category").cat.codes

    ###
    '''from fast ai '''
    from pandas import DataFrame
    import re
    from functools import partial
    import calendar
    from typing import Sequence, Tuple, TypeVar, Union
    def ifnone(a, b):
        "`a` if `a` is not None, otherwise `b`."
        return b if a is None else a

    def make_date(df: DataFrame, date_field: str):
        "Make sure `df[field_name]` is of the right date type."
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

    def cyclic_dt_feat_names(time: bool = True, add_linear: bool = False):
        "Return feature names of date/time cycles as produced by `cyclic_dt_features`."
        fs = ['cos', 'sin']
        attr = [f'{r}_{f}' for r in 'weekday day_month month_year day_year'.split() for f in fs]
        if time: attr += [f'{r}_{f}' for r in 'hour clock min sec'.split() for f in fs]
        if add_linear: attr.append('year_lin')
        return attr

    def cyclic_dt_features(d, time: bool = True, add_linear: bool = False):
        "Calculate the cos and sin of date/time cycles."
        tt, fs = d.timetuple(), [np.cos, np.sin]
        day_year, days_month = tt.tm_yday, calendar.monthrange(d.year, d.month)[1]
        days_year = 366 if calendar.isleap(d.year) else 365
        rs = d.weekday() / 7, (d.day - 1) / days_month, (d.month - 1) / 12, (day_year - 1) / days_year
        feats = [f(r * 2 * np.pi) for r in rs for f in fs]
        if time and isinstance(d, datetime) and type(d) != date:
            rs = tt.tm_hour / 24, tt.tm_hour % 12 / 12, tt.tm_min / 60, tt.tm_sec / 60
            feats += [f(r * 2 * np.pi) for r in rs for f in fs]
        if add_linear:
            if type(d) == date:
                feats.append(d.year + rs[-1])
            else:
                secs_in_year = (datetime(d.year + 1, 1, 1) - datetime(d.year, 1, 1)).total_seconds()
                feats.append(d.year + ((d - datetime(d.year, 1, 1)).total_seconds() / secs_in_year))
        return feats

    def add_cyclic_datepart(df: DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False,
                            add_linear: bool = False):
        "Helper function that adds trigonometric date/time features to a date in the column `field_name` of `df`."
        make_date(df, field_name)
        field = df[field_name]
        prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
        series = field.apply(partial(cyclic_dt_features, time=time, add_linear=add_linear))
        columns = [prefix + c for c in cyclic_dt_feat_names(time, add_linear)]
        df_feats = pd.DataFrame([item for item in series], columns=columns, index=series.index)
        for column in columns: df[column] = df_feats[column]
        if drop: df.drop(field_name, axis=1, inplace=True)
        return df

    def add_datepart(df: DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False):
        '''
          'datetimeweekday_cos',
        'datetimeweekday_sin', 'datetimeday_month_cos', 'datetimeday_month_sin',
        'datetimemonth_year_cos', 'datetimemonth_year_sin',
        'datetimeday_year_cos', 'datetimeday_year_sin'
        Helper function that adds columns relevant to a date in the column `field_name`
        of `df`.
        '''
        make_date(df, field_name)
        field = df[field_name]
        prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
                'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[prefix + n] = getattr(field.dt, n.lower())
        df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
        if drop: df.drop(field_name, axis=1, inplace=True)
        return df

    add_datepart(df, 'datetime', drop=False)  # inplace
    add_cyclic_datepart(df, 'datetime', drop=False) # inplace

    df['title_truncated'] = df['title'].str.split('(', expand=True).loc[:, 0].str.rstrip().str.rstrip('!').str.rstrip(
        '?')
    #is_special_char_mask = df['title_truncated'].apply(lambda d: isStringContainSpecialCharacter(d))
    print(
        f"{len(df['title']) - df['title'].nunique()} raw titles are identical between songs: {df['title'].nunique()} unique titles")
    print(
        f"After cleaning brackets etc. only {df['title_truncated'].nunique()} unique titles remain, i.e. {df['title'].nunique() - df['title_truncated'].nunique()} are highly similar titles ")

    # It seems like all songs on albums release at the same time, so groupby by release_time will create album
    df["album_right"] = df.release_time.astype("category").cat.codes
    df["albumHashAndName"] = df["album_hash"].fillna(df['album_raw_from_mp3_metadata'])
    print(
        f"In making albumHashandName, we filled in: "
        f"{len(set(df[df['album_hash'].isnull()].index) - set(df[df['album_raw_from_mp3_metadata'].isnull()].index))} "
        f"values Of the total albumHashAndName {df['albumHashAndName'].isnull().sum()} nan")
    df["albumHashAndNameAndReleaseday"] = df["albumHashAndName"].fillna(df['album_right'])

    print(
        f"In making albumHashAndNameAndReleaseday, we filled in the: "
        f"{len(set(df[df['albumHashAndName'].isnull()].index))} values remaining using the release second hash")
    assert df['albumHashAndNameAndReleaseday'].isnull().sum() == 0


    import re
    def isContainsSpecialChar(string):
        # Make own character set and pass
        # this as argument in compile method
        regex = re.compile('^.*[^a-zA-Z0-9_]')  # [@_!#$%^&*()<>?/\|}{~:]

        # Pass the string in search
        # method of regex object.
        if (regex.search(string) == None):
            return False

        else:
            return True
    df['title_truncated'] = df['title'].str.split('(', expand=True).loc[:, 0].str.rstrip().str.rstrip('!').str.rstrip(
        '?')
    is_special_char_mask = df['title_truncated'].apply(lambda d: isContainsSpecialChar(d))
    _df_train = df[df.dataset == "train"]
    english_like_names = _df_train.loc[_df_train['title_truncated'][~is_special_char_mask].index]['label']
    test = ttest_ind(_df_train['label'], english_like_names)
    if test.pvalue < 0.05:
        print(
            "There is a statistically signficiant relationship between English-like title and rank. So adding feature: isEnglishLikeTitle")
    df['isEnglishLikeTitle'] = ~is_special_char_mask


    import re
    def get_min_artist_id(s):
        ps = re.split(',|\.', s)
        ps = [int(p) for p in ps]
        return np.min(ps)

    def get_max_artist_id(s):
        ps = re.split(',|\.', s)
        ps = [int(p) for p in ps]
        return np.max(ps)

    df["artist_id_min"] = df["artist_id"].apply(lambda x: get_min_artist_id(x))
    df["artist_id_min_cat"] = df["artist_id_min"].astype('category')
    df["artist_id_min_cat"] = df["artist_id_min_cat"].cat.codes

    df["composers_id_min"] = df["composers_id"].apply(lambda x: get_min_artist_id(x))
    df["composers_id_min_cat"] = df["composers_id_min"].astype('category')
    df["composers_id_min_cat"] = df["composers_id_min_cat"].cat.codes

    df["artist_id_max"] = df["artist_id"].apply(lambda x: get_max_artist_id(x))
    df["artist_id_max_cat"] = df["artist_id_max"].astype('category')
    df["artist_id_max_cat"] = df["artist_id_max_cat"].cat.codes

    df["composers_id_max"] = df["composers_id"].apply(lambda x: get_max_artist_id(x))
    df["composers_id_max_cat"] = df["composers_id_max"].astype('category')
    df["composers_id_max_cat"] = df["composers_id_max_cat"].cat.codes

    df["num_same_title"] = df.groupby("title")["title"].transform("count")
    ##############
    # These use knowledge of entire dataset X values
    ##############

    df["numsongInAlbum"] = df.groupby("albumHashAndNameAndReleaseday")["albumHashAndNameAndReleaseday"].transform("count")
    df["isSingleAlbum_onesong"] = df["isSingleAlbum"] & (df["numsongInAlbum"] == 1)

    '''
    # Find the number of songs which were released between 5-6 months from the datetime field == the release date
    def find_num_song_released_that_week(df, day):
        fromtime = day + relativedelta.relativedelta(days=7)
        totime = day
        return len(df.datetime[(df.datetime >= fromtime) & (df.datetime <= totime)])

    df["num_song_released_that_week"] = df.datetime.apply(lambda d: find_num_song_released_that_week(df, d))
    '''
    # Find the number of songs which were released between 5-6 months from the datetime field == the release date
    def find_num_song_release_in_final_month(df, day):
        month5th = day + relativedelta.relativedelta(months=5)
        month6th = day + relativedelta.relativedelta(months=6)
        return len(df.datetime[(df.datetime >= month5th) & (df.datetime <= month6th)])

    df["num_song_release_in_final_month"] = df.datetime.apply(lambda d: find_num_song_release_in_final_month(df, d))

    df["freq_artist"] = df.groupby('artist_id')['artist_id'].transform('count').astype('float')
    df["freq_composer"] = df.groupby('composers_id')['composers_id'].transform('count').astype('float')


    df["_artist_id_min_cat"] = df["artist_id_min"].astype('category')
    df["_artist_id_min_cat"] = df["_artist_id_min_cat"].cat.codes

    df["_composers_id_min_cat"] = df["composers_id_min"].astype('category')
    df["_composers_id_min_cat"] = df["_composers_id_min_cat"].cat.codes


    df["freq_artist_min"] = df.groupby('_artist_id_min_cat')['_artist_id_min_cat'].transform('count').astype('float')
    df["freq_composer_min"] = df.groupby('_composers_id_min_cat')['_composers_id_min_cat'].transform('count').astype(
        'float')
    df["num_album_per_min_artist"] = df.groupby(['_artist_id_min_cat', 'albumHashAndNameAndReleaseday'])['albumHashAndNameAndReleaseday'].transform('count').astype(
        'float')
    df["num_album_per_min_composer"] = df.groupby(['composers_id_min', 'albumHashAndNameAndReleaseday'])['albumHashAndNameAndReleaseday'].transform('count').astype(
        'float')

    df = df.drop(['album', 'album_hash'], axis = 1)

    return df


def baysianEncodeFeature(df_train, trn_idx, featurename, prior_weight, fillmissing, suffix='_baysencoded'):
    '''Returns new df '''
    import xam

    encoder = xam.feature_extraction.BayesianTargetEncoder(
        columns=[featurename, ],
        prior_weight=prior_weight,
        suffix=suffix)

    train_df_fold = df_train.iloc[trn_idx]

    encoder.fit(train_df_fold[[featurename]], train_df_fold.label)

    _resulting_df = encoder.transform(df_train[[featurename]], df_train.label)
    _resulting_df[featurename + suffix] = _resulting_df[featurename + suffix].astype('float64')
    _resulting_df[featurename + suffix].fillna(fillmissing, inplace=True)

    # Add the column to original df_train
    df_train[featurename + suffix] = _resulting_df[featurename + suffix]#.round(0).astype('int64')
    return df_train


from functools import reduce


def create_album_score_lookup_table(df):
    data = df.groupby('album_right').label.agg(["mean", "std", "count"])
    return data


def create_artist_score_lookup_table(df):
    def split_id(s):
        return re.split(',|\.', s)

    def mask_row_contain_artist_id(df, artist_id):
        r = df.artist_id.apply(lambda x: artist_id in split_id(x))
        return r

    # Get all artist ids
    artist_group = df.artist_id.unique()
    artist_ids = reduce(lambda l, e: l + split_id(e), artist_group, [])
    artist_ids = list(set(artist_ids))
    # Get data
    data = [df[mask_row_contain_artist_id(df, artist_id)].label.agg(["mean", "std", "count", "median"])
            for artist_id in artist_ids]
    new_df = pd.DataFrame(data=data)
    new_df["artist_id"] = artist_ids
    return new_df.set_index("artist_id")


def create_album_score_lookup_table(df):
    data = df.groupby('album_right').label.agg(["mean", "std", "count"])
    return data

def create_artist_score_lookup_table(df):
    def split_id(s):
        return re.split(',|\.', s)

    def mask_row_contain_artist_id(df, artist_id):
        r = df.artist_id.apply(lambda x: artist_id in split_id(x))
        return r

    # Get all artist ids
    artist_group = df.artist_id.unique()
    artist_ids = reduce(lambda l, e: l + split_id(e), artist_group, [])
    artist_ids = list(set(artist_ids))
    # Get data
    data = [df[mask_row_contain_artist_id(df, artist_id)].label.agg(["mean", "std", "count", "median"])
            for artist_id in artist_ids]
    new_df = pd.DataFrame(data=data)
    new_df["artist_id"] = artist_ids
    return new_df.set_index("artist_id")

def get_field_by_key(table, k, field="mean"):
    if k in table.index:
        return table.loc[k][field]
    return np.nan

def get_value_by_key(table, k):
    if k in table.index:
        return table.loc[k], False
    return np.nan, True

def assign_value(album_table, artist_table, r):
    d1, isnul1 = get_value_by_key(album_table, r.album_right)
    d2, isnul2 = get_value_by_key(artist_table, r.artist_id_min_cat)
    #     print(type(d2),isnul2)
    if isnul1 and isnul2:
        return np.nan
    elif isnul1 and d2["std"] < 2:
        return d2["mean"]
    elif isnul2 and d1["std"] < 2:
        return d1["mean"]

    elif not isnul1 and d1["std"] < 2 and not isnul2 and d2["std"] < 2:
        return d1["mean"]

    return np.nan

def assign_artist_features_inplace(df):

    def split_id(s):
        return re.split(',|\.', s)

    m = df.artist_id.unique()
    idx_lst = []
    for idx in m:
        ps = split_id(idx)
        for i in ps:
            idx_lst.append(i)

    id_lst = list(set(idx_lst))

    def condition(df, artist_id):
        r = df.artist_id.apply(lambda x: artist_id in split_id(x))
        return r

    df_train = df[df.dataset == "train"]
    data = [df_train[condition(df_train, artist_id)].label.agg(["mean", "std", "count"]) for artist_id in id_lst]
    new_df = pd.DataFrame(data=data)
    new_df["artist_id"] = id_lst

    new_df.dropna(inplace=True)
    new_df.set_index('artist_id', inplace=True)
    art_dict = new_df.to_dict()

    def best_count_id(values):
        ids = split_id(values)
        temp_mean = 0
        temp_id = str(min([int(a) for a in ids]))
        for id in ids:
            try:
                if art_dict['count'][id] > temp_mean:
                    temp_mean = art_dict['count'][id]
                    temp_id = id
            except:
                temp_mean = temp_mean
                temp_id = temp_id
        return temp_id

    def best_mean_id(values):
        ids = split_id(values)
        temp_mean = 10
        temp_id = str(min([int(a) for a in ids]))
        for id in ids:
            try:
                if art_dict['mean'][id] < temp_mean:
                    temp_mean = art_dict['mean'][id]
                    temp_id = id
            except:
                temp_mean = temp_mean
                temp_id = temp_id
        return temp_id

    def best_std_id(values):
        ids = split_id(values)
        temp_std = 10
        temp_id = str(min([int(a) for a in ids]))
        for id in ids:
            try:
                if art_dict['std'][id] < temp_std:
                    temp_std = art_dict['std'][id]
                    temp_id = id
            except:
                temp_std = temp_std
                temp_id = temp_id
        return temp_id

    df['artist_count_id'] = df['artist_id'].apply(best_count_id)
    df['artist_mean_id'] = df['artist_id'].apply(best_mean_id)
    df['artist_std_id'] = df['artist_id'].apply(best_std_id)
    df['artist_mean_id'] = df['artist_mean_id'].astype("category")
    df['artist_std_id'] = df['artist_std_id'].astype("category")
    df['artist_count_id'] = df['artist_count_id'].astype("category")

    return df
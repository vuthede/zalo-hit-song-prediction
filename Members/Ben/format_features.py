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

def format_features(df):
    # Fill nan album
    print("There is {} ratio is nan album".format(len(df[df["album"].isnull()]) / len(df)))
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

    df["artist_name"] = df["artist_name"].astype('category')
    df["composers_name"] = df["composers_name"].astype('category')
    df["copyright"] = df["copyright"].astype('category')


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
    df["artist_id_max"] = df["artist_id"].apply(lambda x: get_max_artist_id(x))
    df["composers_id_min"] = df["composers_id"].apply(lambda x: get_min_artist_id(x))
    df["composers_id_max"] = df["composers_id"].apply(lambda x: get_max_artist_id(x))

    # New feature
    # df["group_album_artist_id_min_cat"] = df.groupby(["album","artist_id_min_cat"]).ngroup()
    # df["group_album_artist_id_min_cat"] = df["group_album_artist_id_min_cat"].astype("category").cat.codes
    # df["group_album_artist_id_max_cat"] = df.groupby(["album","artist_id_max_cat"]).ngroup()
    # df["group_album_artist_id_max_cat"] = df["group_album_artist_id_max_cat"].astype("category").cat.codes

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

    def add_datepart(df: DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False):
        "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
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
    ###

    ##############
    # These use knowledge of entire dataset X values
    ##############
    # It seems like all songs on albums release at the same time, so groupby by release_time will create album
    df["album_right"] = df.groupby(df.release_time).ngroup().astype("category").cat.codes
    df["numsongInAlbum"] = df.groupby("album_right")["album_right"].transform("count")
    df["isSingleAlbum_onesong"] = df["isSingleAlbum"] & (df["numsongInAlbum"] == 1)

    # Find the number of songs which were released between 5-6 months from the datetime field == the release date
    def find_num_song_released_that_week(df, day):
        fromtime = day + relativedelta.relativedelta(days=7)
        totime = day
        return len(df.datetime[(df.datetime >= fromtime) & (df.datetime <= totime)])

    df["num_song_released_that_week"] = df.datetime.apply(lambda d: find_num_song_released_that_week(df, d))

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
    df["num_album_per_min_artist"] = df.groupby(['_artist_id_min_cat', 'album'])['album'].transform('count').astype(
        'float')
    df["num_album_per_min_composer"] = df.groupby(['composers_id_min', 'album'])['album'].transform('count').astype(
        'float')

    return df
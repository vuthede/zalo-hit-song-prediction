
import pandas as pd
def add_raw_audio_features(df, all_features_in_order):
    not_included_columns = set(df.columns) - set([i for i in all_features_in_order.keys()]) - set(["tonal.chords_key", "tonal.chords_scale"])
    audio_feature_names = sorted([i for i in not_included_columns if i.split(".")[0] == 'tonal'
            or  i.split(".")[0] == 'rhythm'
            or  i.split(".")[0]=='lowlevel'])
    #pd.set_option('display.max_columns', 500)
    audio_feature_names_dict = {name:"float64" for name in audio_feature_names}
    audio_feature_names_dict["tonal.chords_key"] = "category"
    audio_feature_names_dict["tonal.chords_scale"] = "category"
    audio_feature_names_dict["tonal.key_edma.key"] = "category"
    audio_feature_names_dict["tonal.key_edma.scale"] = "category"
    audio_feature_names_dict["tonal.key_krumhansl.key"] = "category"
    audio_feature_names_dict["tonal.key_krumhansl.scale"] = "category"
    audio_feature_names_dict["tonal.key_temperley.key"] = "category"
    audio_feature_names_dict["tonal.key_temperley.scale"] = "category"
    all_features_in_order.update(audio_feature_names_dict)
    return all_features_in_order

def typecast_features(df, cast_to_catcode=True):
    all_features_in_order = {
        ## album is Redundant feature use: albumHashAndNameAndReleaseday", # album name from mp3 metadata textual
        "albumHashAndNameAndReleaseday": "category",
        "len_album_name": "int64",
        "isRemixAlbum": "bool",
        "isOSTAlbum": "bool",
        "isSingleAlbum": "bool",
        "isBeatAlbum": "bool",
        "isTopHitAlbum": "bool",
        "isCoverAlbum": "bool",
        "isEPAlbum": "bool",
        "isLienKhucAlbum": "bool",
        "album_name_is_title_name": "category",
        "artist_name": "category",
        "composers_name": "category",
        "copyright": "category",
        "artist_id_min": "category",
        "artist_id_max": "category",
        "composers_id_min": "category",
        "composers_id_max": "category",
        "genre": "category",
        "album_artist": "category",  # album artist name
        "album_artist_contain_artistname": "category",
        "track": "float64",  # float between 0 and 1 representing track_num/total_tracks
        "istrack11": "bool",  # 1 if first track
        # "lyric":"string" # Not a trainable feature
        "islyric": "bool",
        "num_line_lyric": "int64",
        "no_artist": "int64",
        "no_composer": "int64",
        # "datetime":"datetime64", # Not a trainable feature
        "day": "category",
        "month": "category",
        "year": "category",
        "hour": "category",
        "dayofyear": "int64",
        "weekday": "category",
        "isHoliday": "category",
        "len_of_songname": "int64",
        "isRemix": "bool",
        "isOST": "bool",
        "isBeat": "bool",
        "isVersion": "bool",
        "isCover": "bool",
        "isLienKhuc": "bool",
        "day_release": "int64",  # the specific day of the day across all days (> 365)
        "datetimeIs_month_end": "category",
        "datetimeIs_month_start": "category",
        "datetimeIs_quarter_end": "category",
        "datetimeIs_quarter_start": "category",
        "datetimeIs_year_end": "category",
        "datetimeIs_year_start": "category",
        "datetimeDayofweek": "category",
        "tonal.chords_key": "category",
        "tonal.chords_scale": "category",
        'datetimeweekday_cos': "float64",
        'datetimeweekday_sin': "float64",
        'datetimeday_month_cos': "float64",
        'datetimeday_month_sin': "float64",
        'datetimemonth_year_cos': "float64",
        'datetimemonth_year_sin': "float64",
        'datetimeday_year_cos': "float64",
        'datetimeday_year_sin': "float64",
        "length": "float64",  # length of the song (s?)
        # (title feature likely rendered redundant - use title_truncated )"title":"category",
        'title_truncated': "category",
        "isEnglishLikeTitle": "bool",
        ###########################
        # Warning: the below features those that require "global" knowledge beyond that example
        ###########################
        "numsongInAlbum": "category",
        "isSingleAlbum_onesong": "bool",
        # "num_song_released_that_week":'int64',
        "num_song_release_in_final_month": "int64",
        "freq_artist": "int64",  # number of times the unique artist string is present in dataset
        "freq_artist_min": "int64",  # number of times the first listed artist is present in dataset.  # floats in original
        "num_album_per_min_artist": "int64",
        "num_album_per_min_composer": "int64",
        "freq_composer_min": "int64",

        # has some words with no special characters - in practice selects short and weird titles
    }
    all_features_in_order = add_raw_audio_features(df, all_features_in_order)
    all_features_in_order_list = list(all_features_in_order.keys())
    for feat_name, feat_type in all_features_in_order.items():
        try:
            df[feat_name] = df[feat_name].astype(feat_type)
        except ValueError:
            print(feat_name, feat_type)
            raise
    if cast_to_catcode:
        df = cast_cat_dtype_to_cat_codes(df, all_features_in_order_list)

    return all_features_in_order_list, df

def cast_cat_dtype_to_cat_codes(df, column_names):
    for colname in df[column_names].select_dtypes(include=['category']).columns :
        df[colname] = df[colname].cat.codes
        # Lightgbm also assumes integer casting is provided
    return df
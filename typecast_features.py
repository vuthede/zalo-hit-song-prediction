
import pandas as pd


def typecast_features(df, cast_to_catcode=True):
    all_features_in_order = {
        "artist_name": "category",
        "composers_name": "category",
        "copyright": "category",
        "genre": "category",
        "album_artist": "category",  # album artist name
        "track": "float64",  # float between 0 and 1 representing track_num/total_tracks
        'title_truncated': "category",
        "isEnglishLikeTitle": "bool",
        "numsongInAlbum": "int64",
        "isSingleAlbum_onesong": "bool",
        "title_cat":"category",
    }
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
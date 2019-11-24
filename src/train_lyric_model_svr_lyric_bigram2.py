import pandas as pd
import numpy as np
import random
from pyvi import ViTokenizer, ViPosTagger
from pyvi import ViUtils
from nltk.tokenize import word_tokenize
import nltk
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion, make_union
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer

def rmse(targets, predictions):
    return np.sqrt(mean_squared_error(targets, predictions))
np.random.seed(1)
random.seed(1)

TRAININFO = "/media/DATA/zalo-hit-song-prediction/csv/train_info.tsv"
TRAINRANK =  "/media/DATA/zalo-hit-song-prediction/csv/train_rank.csv"
TESTINFO = "/media/DATA/zalo-hit-song-prediction/csv/test_info.tsv"
Track_info = "/media/DATA/zalo-hit-song-prediction/csv/all_track_info.csv"
Audio_info = "/media/DATA/zalo-hit-song-prediction/csv/all_track_audio_features.csv"
df_i = pd.read_csv(TRAININFO, delimiter='\t',encoding='utf-8')
df_r = pd.read_csv(TRAINRANK)
df_i_train = df_i.merge(df_r, left_on='ID', right_on='ID')
df_i_train["dataset"] = "train"

df_i_test = pd.read_csv(TESTINFO, delimiter='\t',encoding='utf-8')
df_i_test["label"] = np.nan
df_i_test["dataset"] = "test"

df = pd.concat([df_i_train, df_i_test])
df_track_info = pd.read_csv(Track_info)
df = df.merge(df_track_info, left_on='ID', right_on='ID')
df_audio_features = pd.read_csv(Audio_info)
df =df.merge(df_audio_features,left_on="ID",right_on="ID", how="left")
df = df[['ID','title','artist_name','lyric','label','dataset']]
df.head()
'''
def encode_column(df, column_name, to_n_columns=12):

    hasher=category_encoders.hashing.HashingEncoder()
    #encoded=hasher.fit_transform(np.expand_dims(df["university_name"].values, axis=-1),verbose=True, cols=None, drop_invariant=True, return_df=True, hash_method='sha1')
    tmp=pd.DataFrame(np.expand_dims(df["university_name"].values, axis=-1))
    encoded_df=hasher.hashing_trick(tmp, hashing_method='md5', N=to_n_columns, make_copy=False)

    encoded_col_names=list(encoded_df.columns.values)
    tmp=df.copy()
    tmp["index"]=tmp.index
    tmp=tmp.reset_index()
    df=pd.merge(tmp, encoded_df, how='left', left_index=True, right_index=True, sort=False,
             suffixes=('_left', '_right'),validate="one_to_one")

    print(type(encoded_col_names))
    df.set_index("index",inplace=True)
    return df, encoded_col_names
'''

class TextColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

with open("/media/DATA/zalo-hit-song-prediction/csv/stopwords/vietnamese-stopwords.txt", 'r') as f:
    filecontent=f.readlines()
stopwords = list(set([f.strip() for f in filecontent]))
stopwords_tokenized = list(set([ViTokenizer.tokenize(stopword.lower()) for stopword in stopwords]))
set([ViTokenizer.tokenize(word) for word in stopwords_tokenized]) - set(stopwords_tokenized)
df.dropna(subset=['lyric'], inplace=True) # drop songs with no lyrics
df_train = df[df.dataset=='train']
df_test = df[df.dataset=='test']
df_train = df_train.drop_duplicates(subset='lyric', keep='first')

# generate features - in this case just one.
feature = Pipeline([
    ('selector_1', TextColumnSelector(key='lyric')),
    ('vect', HashingVectorizer(n_features=1500,
                               tokenizer=ViTokenizer.tokenize,
                               lowercase=True,
                               stop_words=stopwords_tokenized,
                               norm='l2')),
]
    # ('vect', CountVectorizer(analyzer='word', tokenizer=str.split, stop_words=stopwords)),
    # ('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=True))
)
# Another option:
##hasher=category_encoders.hashing.HashingEncoder()
##encoded_df=hasher.hashing_trick(tmp, hashing_method='md5', N=to_n_columns, make_copy=False)
# vectorizer = HashingVectorizer(n_features = 2**4, tokenizer=word_tokenize, lowercase=True, stop_words=stopwords,  norm='l2')
# vectors = vectorizer.fit_transform(df_train.lyric)

feats_list = [('text', feature)]
feats = FeatureUnion(feats_list)
rmse_scorer = make_scorer(rmse, greater_is_better=False)
#model=LogisticRegression(class_weight="balanced",penalty='l1', random_state=1)
#model_lc="classifier__C"
#values=[719.686, ] # np.logspace(-4, 4, 8) # 0.001 to 1e4
model2=SVR(kernel="rbf", C=0.615848, gamma=11.2884)

model_pipeline = Pipeline([
    ('features',feats),
    ('classifier', model2),
])
'''
model_lc="classifier__gamma"
values = [11.2884,]
model_lc_2 = "classifier__C"
values2 = [0.615848,]

model_pipeline = Pipeline([
    ('features',feats),
    ('classifier', model),
])

#param_range = np.logspace(1, 10, 5)

parameters = {'features__text__vect__n_features': np.array([1500,]).astype('int64'),
              'features__text__vect__ngram_range': [(1,2),],# [(1,1), (1, 2),], # try bigrams or unigrams
              model_lc: values,
              model_lc_2:values2,
             }

skf = StratifiedKFold(n_splits=10, random_state=99999)
gs_clf = RandomizedSearchCV(model_pipeline,
                      parameters,
                      scoring=rmse_scorer,
                      cv=skf,
                      n_jobs=-1,
                      return_train_score=False,
                      error_score='raise',
                      n_iter=100,
                      verbose=10
                    )

#gs_clf.get_params().keys()
#[i for i in gs_clf.get_params().keys()]
gs_clf.fit(df_train, df_train.label)

print(gs_clf.cv_results_['mean_test_score'][gs_clf.best_index_], "+/-" ,gs_clf.cv_results_['std_test_score'][gs_clf.best_index_])
print(gs_clf.best_score_, gs_clf.best_params_)
pd.DataFrame(gs_clf.cv_results_).to_csv('svr_lyric_bigram2.csv')
'''
model_pipeline.fit(df_train, df_train.label)
joblib.dump(model2, 'svr_lyric_bigram2_trained_on_whole_set.pkl')
preds = model_pipeline.predict(df_train)
df_train['lyric_feature'] = preds
df_train[['ID', 'lyric_feature']].to_csv('../csv/lyric_feature.csv')
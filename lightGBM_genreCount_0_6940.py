from sklearn import ensemble
import pandas as pd
import sys, gc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import lightgbm as lgb
import matplotlib.pyplot as plt

#data_path = "/home/ali/Desktop/data science project/data/"
data_path = '/export/home/adrem01/DataScienceProject/data/'

def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';']))

def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0

def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')



print("data loading ...")
members = pd.read_csv(data_path + 'members.csv',
                      dtype = {'msno'          : 'category',
                               'city'          : 'category',
                               'bd'            : np.uint8,
                               'gender'        : 'category',
                               'registered_via': 'category'},
                      parse_dates = ['registration_init_time',
                                     'expiration_date'])
# Get mean age for users between 12 and 100
mean_age = int(members.loc[members.bd >= 12].loc[members.bd <= 100]['bd'].mean())
# Set age to mean for users with nonsensical age
members['bd'] = members['bd'].apply(lambda age: mean_age if age < 12 or 100 < age else age)
# Encode unknown gender instead of NaN
members['gender'].cat.add_categories('unknown', inplace = True)
members['gender'].fillna('unknown', inplace = True)
# Compute member lifetime in days
day_ns = 24 * 60 * 60 * pow(10,9)
members['lifetime'] = pd.to_numeric(members['expiration_date'] - members['registration_init_time']) / day_ns
# Drop date columns, can only lead to overfitting
members.drop(['registration_init_time','expiration_date'], axis = 1, inplace = True)


songs = pd.read_csv(data_path + 'songs_extra_cols_null_values.csv',
                    dtype = {'song_id'    : 'category',
                             'song_length': np.uint32,
                             'genre_ids'  : 'category',
                             'artist_name': 'category',
                             'composer'   : 'category',
                             'lyricist'   : 'category',
                             'language'   : 'category',
                             'genre_0'    : 'category',
                             'genre_1'    : 'category',
                             'genre_2'    : 'category',
                             'genre_3'    : 'category',
                             'genre_4'    : 'category',
                             'genre_5'    : 'category',
                             'genre_6'    : 'category',
                             'genre_7'    : 'category'

                            })


train = pd.read_csv(data_path + 'train.csv',
                    dtype = {'msno': 'category',
                             'song_id': 'category',
                             'source_system_tab': 'category',
                             'source_screen_name': 'category',
                             'source_type': 'category'})
test  = pd.read_csv(data_path + 'test.csv',
                    dtype = {'msno': 'category',
                             'song_id': 'category',
                             'source_system_tab': 'category',
                             'source_screen_name': 'category',
                             'source_type': 'category'})



# Fill NaN values
for column in ['source_system_tab', 'source_screen_name', 'source_type']:
    train[column].cat.add_categories('no_' + column, inplace = True)
    test[column].cat.add_categories('no_' + column, inplace = True)
    train[column].fillna('no_' + column, inplace = True)
    test[column].fillna('no_' + column, inplace = True)

train = pd.merge(pd.merge(train, members, on = 'msno', how = 'left'), songs, on = 'song_id', how = 'left')
test  = pd.merge(pd.merge(test, members, on = 'msno', how = 'left'), songs, on = 'song_id', how = 'left')


# Fill NaNs for songs
for column in ['genre_0','genre_1','genre_2','genre_3','genre_4','genre_5','genre_6','genre_7', 'language']:
    train[column].cat.add_categories('no_' + column, inplace=True)
    test[column].cat.add_categories('no_' + column, inplace=True)
    train[column].fillna('no_' + column, inplace = True)
    test[column].fillna('no_' + column, inplace = True)

# Fill song length with mean
train['song_length'].fillna(songs['song_length'].mean(), inplace = True)
test['song_length'].fillna(songs['song_length'].mean(), inplace = True)

song_extra_info = pd.read_csv(data_path + "song_extra_info.csv",
                              dtype= {
                                  'song_id': 'category',
                                  'name': 'category',
                                  'isrc': 'category'})

song_extra_info['name'].cat.add_categories('no_' + 'name', inplace = True)
song_extra_info['name'].fillna('no_' + 'name', inplace = True)

song_extra_info['CC'] = song_extra_info['isrc'].apply(lambda x: (str(x)[0:2]))
song_extra_info['CC'] = song_extra_info['CC'].astype('category')
song_extra_info['Issuer'] = song_extra_info['isrc'].apply(lambda x: (str(x)[2:5]))
song_extra_info['Issuer'] = song_extra_info['Issuer'].astype('category')

song_extra_info['isrc_year'] = song_extra_info['isrc'].apply(lambda x:  (str(x)[5:7]))
song_extra_info['isrc_year'] = song_extra_info['isrc_year'].astype('category')


song_extra_info.drop(['isrc'], axis = 1, inplace = True)
train = pd.merge(train, song_extra_info, on = 'song_id', how = 'left')
test  = pd.merge(test, song_extra_info, on = 'song_id', how = 'left')

del songs;del members;del song_extra_info;gc.collect()

print("adding new features ...")


#train['play_count_artist'] = train.groupby(['artist_name'])['artist_name'].transform('count')
#test['play_count_artist'] = test.groupby(['artist_name'])['artist_name'].transform('count')

print(train.head(5))
print(train.shape)
print(train.columns)
train['genre_ids'].cat.add_categories('no_genre_id', inplace = True)
test['genre_ids'].cat.add_categories('no_genre_id', inplace = True)
train['genre_ids'].fillna('no_genre_id',inplace=True)
test['genre_ids'].fillna('no_genre_id',inplace=True)
train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int64)
test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int64)

train['lyricist'].cat.add_categories('no_lyricist', inplace = True)
test['lyricist'].cat.add_categories('no_lyricist', inplace = True)
train['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int64)
#train['lyricists_count'] = train.groupby('lyricist')['lyricist'].transform('count').astype(np.int64)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int64)
#test['lyricists_count'] = test.groupby('lyricist')['lyricist'].transform('count').astype(np.int64)
train['composer'].cat.add_categories('no_composer', inplace = True)
test['composer'].cat.add_categories('no_composer', inplace = True)
train['composer'].fillna('no_composer',inplace=True)
test['composer'].fillna('no_composer',inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int64)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int64)

train['artist_name'].cat.add_categories('no_artist', inplace = True)
test['artist_name'].cat.add_categories('no_artist', inplace = True)
train['artist_name'].fillna('no_artist',inplace=True)
test['artist_name'].fillna('no_artist',inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int64)
test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int64)

# if artist is same as composer
train['artist_composer'] = (np.asarray(train['artist_name']) == np.asarray(train['composer'])).astype(np.int64)
test['artist_composer'] = (np.asarray(test['artist_name']) == np.asarray(test['composer'])).astype(np.int64)


# if artist, lyricist and composer are all three same
train['artist_composer_lyricist'] = ((np.asarray(train['artist_name']) == np.asarray(train['composer'])) & (np.asarray(train['artist_name']) == np.asarray(train['lyricist'])) & (np.asarray(train['composer']) == np.asarray(train['lyricist']))).astype(np.int64)
test['artist_composer_lyricist'] = ((np.asarray(test['artist_name']) == np.asarray(test['composer'])) & (np.asarray(test['artist_name']) == np.asarray(test['lyricist'])) & (np.asarray(test['composer']) == np.asarray(test['lyricist']))).astype(np.int64)

train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int64)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int64)


train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int64)
test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int64)

_mean_song_length = np.mean(train['song_length'])
def smaller_song(x):
    if x < _mean_song_length:
        return 1
    return 0

train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int64)
test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int64)

# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}


def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0


train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)

# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}


def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0


train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)


#adding cluster feature.
cluster = pd.read_csv(data_path + 'cluster_songs_20.csv',
                      dtype={'id': np.int8,
                             'song_id': 'category',
                             'cluster':'category'
                      })
cluster.drop('id', axis=1, inplace=True)


train = pd.merge(train, cluster, on='song_id', how='left')
test = pd.merge(test,cluster,on='song_id', how='left')

train['cluster'].cat.add_categories('no_cluster', inplace = True)
train['cluster'].fillna('no_cluster',inplace=True)
test['cluster'].cat.add_categories('no_cluster', inplace = True)
test['cluster'].fillna('no_cluster',inplace=True)

print('Done adding features')
print("data labeling ....")

# Merge genre counts
genreCounts = pd.read_csv('genreCounts.csv',
                            dtype={ 'genre_ids'  : 'category',
                            'msno'  : 'category'})
genreCounts.drop('Unnamed: 0', axis=1, inplace=True)
genreCounts['genre_ids'].cat.add_categories('no_genre_id', inplace = True)
genreCounts['genre_ids'].fillna('no_genre_id',inplace=True)

train = train.merge(genreCounts, how="left", on=["msno", "genre_ids"])
train['genre_counts'] =train['genre_counts'].fillna(0)
train['genre_ids'] =train['genre_ids'].astype('category')
try:
    train.drop('Unnamed: 0', axis=1, inplace=True)
    print('----------------')
    print(train.head(5))
    print(train.columns)
    print(train.shape)
except:
    pass

test = test.merge(genreCounts, how="left", on=["msno", "genre_ids"])
test['genre_counts'] = test['genre_counts'].fillna(0)
test['genre_ids'] =test['genre_ids'].astype('category')
try:
    test.drop('Unnamed: 0', axis=1, inplace=True)
    print(test.head(5))
    print(test.columns)
    print(test.shape)
except:
    pass

for col in train.columns:
    if str(train[col].dtype) == 'category':
        print('Encoding {0}...'.format(col))
        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

for col in ['msno', 'song_id']:
    print('Encoding {0}...'.format(col))
    le = LabelEncoder()
    train_vals = list(train[col].unique())
    test_vals = list(test[col].unique())
    le.fit(train_vals + test_vals)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


exclude = ['target', 'id']
features = [c for c in train.columns if c not in exclude]
test['target'] = 0.0

print(train['genre_ids'])
print(train['genre_ids'].dtype)
print("training .....")
for seed in range(10):
    print('-------------------------')
    print('-------- MODEL {0} ------'.format(seed))
    print('-------------------------')
    # Set seed for repeatability of experiments
    # seed = 42

    # Set up LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'learning_rate': .4,
        'verbose': 0,
        'num_leaves': 2 ** 10,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': seed,
        'feature_fraction': 0.95,
        'feature_fraction_seed': seed,
        'max_bin': 256,
        'max_depth': 15,
        'num_rounds': 300,
        'early_stopping_round': 10
    }

    # Train - Validation split
    X_train, X_valid, y_train, y_valid = train_test_split(train[features], train['target'], test_size=0.2,
                                                          random_state=seed)

    # Prepare data
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    # Train model
    model = lgb.train(params, lgb_train, verbose_eval=20, valid_sets=[lgb_train, lgb_valid])

    # Generate predictions
    test['target'] += 0.1 * model.predict(test[features])

    test[['id', 'target']].to_csv('predictions_lgbm.csv', index=False)

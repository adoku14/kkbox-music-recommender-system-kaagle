import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
import gc

songs = pd.read_csv('/home/ali/Desktop/data science project/data/songs.csv',
                    dtype = {'song_id'    : 'category',
                             'song_length': np.uint32,
                             'genre_ids'  : 'category',
                             'artist_name': 'category',
                             'composer'   : 'category',
                             'lyricist'   : 'category',
                             'language'   : np.float64

                            })

song_extra_info = pd.read_csv("/home/ali/Desktop/data science project/data/song_extra_info.csv",
                              dtype= {
                                  'song_id': 'category',
                                  'name': 'category',
                                  'isrc': 'category'})

#filling null values.
songs['song_length'].fillna(songs['song_length'].mean(), inplace = True)
songs['language'].fillna(songs['language'].median(), inplace = True)

for column in ['genre_ids', 'artist_name', 'composer', 'lyricist']:
    songs[column].cat.add_categories('unknown' + column, inplace=True)
    songs[column].fillna('unknown' + column, inplace = True)

song_extra_info['name'].cat.add_categories('unknown' + 'name', inplace = True)
song_extra_info['name'].fillna('unknown' + 'name', inplace = True)

song_extra_info['CC'] = song_extra_info['isrc'].apply(lambda x: (str(x)[0:2]))
song_extra_info['CC'] = song_extra_info['CC'].astype('category')
song_extra_info['Issuer'] = song_extra_info['isrc'].apply(lambda x: (str(x)[2:5]))
song_extra_info['Issuer'] = song_extra_info['Issuer'].astype('category')

song_extra_info['isrc_year'] = song_extra_info['isrc'].apply(lambda x:  (str(x)[5:7]))
song_extra_info['isrc_year'] = song_extra_info['isrc_year'].astype('category')

song_extra_info.drop(['isrc'], axis = 1, inplace = True)

songs = pd.merge(songs, song_extra_info, on = 'song_id', how = 'left')

del song_extra_info; gc.collect()
print("data labeling ....")

for col in songs.columns:
    if str(songs[col].dtype) == 'category':
        print('Encoding {0}...'.format(col))
        le = LabelEncoder()
        train_vals = list(songs[col].unique())
        le.fit(train_vals)
        songs[col] = le.transform(songs[col])

label = LabelEncoder()
for col in ["song_id"]:

    print('Encoding {0}...'.format(col))
    train_vals = list(songs[col].unique())
    label.fit(train_vals)
    songs[col] = label.transform(songs[col])

songs = songs.astype(np.int64)

print(songs.head())

print ("starting clustering ...................")
estimator = KMeans(n_clusters=20)
est = estimator.fit(songs)



songs['song_id'] = label.inverse_transform(songs["song_id"])
df_ = pd.DataFrame()
df_['song_id'] = songs['song_id']
df_['cluster'] = est.labels_

df_.to_csv("cluster_songs_20.csv", header=True, sep=',')

#Todo
#1- calculate a probability for each user in cluster and add as feature.
#2- test this way.
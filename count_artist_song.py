import numpy as np
import pandas as pd
import sys, gc, csv
import time

print('Loading data...')
#data_path = 'data/'
data_path = "/home/ali/Desktop/data science project/data/"

# train file is changed adding id as index which will be used later to be merged, according id
#this way it is used, to add for every song/artist the frequency of a song/artist in every iteration.
train = pd.read_csv('train.csv',
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

print('merging .....')
train = pd.merge(train, songs, on = 'song_id', how = 'left')
test  = pd.merge(test, songs, on = 'song_id', how = 'left')

del songs;gc.collect()

print('counting .....')
song_unique = train['song_id'].unique()
artist_played = dict()
song_played = dict.fromkeys(song_unique, 0)



#this function calculate the frequency of artists.
def count_artist_count(df, k):
    cols_a = ['id', 'artist_played', 'ratio_artist_played']
    count_dict_a = dict([(key, []) for key in cols_a])
    i = k  # keep track of the of number of iteration
    for row in zip(df['artist_name']):
        i += 1 # keep track of the of number of iteration
        artist = row[0]
        # if artist is in dict increment its occurence, else add it to dict with value 1.
        if artist in artist_played:
            artist_played[artist] += 1
        else:
            artist_played[artist] = 1
        #save every row with an id which later will be merged according id to train.csv file.
        count_dict_a['id'].append(i - 1 - k)
        count_dict_a['artist_played'].append(artist_played[artist])
        count_dict_a['ratio_artist_played'].append((artist_played[artist] / i))
        # df.ix[[i - 1], 'artist_count'] = artist_played[artist]
        # df.ix[[i - 1], 'ratio_artist_count'] = artist_played[artist] / i
        if i % 100000 == 0:
            print('step: ' + str(i))
    df = pd.DataFrame(count_dict_a)
    df = df[['id', 'artist_played', 'ratio_artist_played']]
    return df, i



#this function calculate the frequency of songs.
def count_songs_played(df, k):
    cols = ['id', 'song_played', 'ratio_song_played']
    count_dict = dict([(key, []) for key in cols])

    i = k  # keep track of the of number of iteration

    for row in zip(df['song_id']):
        i += 1
        song = row[0]
        if song in song_played:
            song_played[song] += 1
        else:
            song_played[song] = 1
        count_dict['id'].append(i - 1 - k)
        count_dict['song_played'].append(song_played[song])
        count_dict['ratio_song_played'].append((song_played[song]/i))
        # df.ix[[i - 1], 'song_played'] = song_played[song]
        # df.ix[[i - 1], 'ratio_song_played'] = song_played[song] / i
        if i % 100000 == 0:
            print('step' + str(i))
    #creating a dataframe to merge later on.
    df = pd.DataFrame(count_dict)
    df = df[['id', 'song_played', 'ratio_song_played']]

    return df, i

start_time = time.time()
print('Counting Songs .....')

train_frame, i = count_songs_played(train, 0)
test_frame, i = count_songs_played(test, i)

print('Counting artists .....')

train_a_frame, i = count_artist_count(train, 0)
test_a_frame, i = count_artist_count(test, i)

print('data merging ......')
train = pd.merge(train, train_frame, on = 'id', how = 'left')
test  = pd.merge(test, test_frame, on = 'id', how = 'left')

train = pd.merge(train, train_a_frame, on = 'id', how = 'left')
test  = pd.merge(test, test_a_frame, on = 'id', how = 'left')

train.drop('id', axis = 1, inplace = True)

train.to_csv('data/train_count.csv', index=False)
test.to_csv('data/test_count.csv', index=False)
print("--- %s seconds ---" % (time.time() - start_time))

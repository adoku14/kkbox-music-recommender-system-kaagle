
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import sys, gc, csv
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter

print('Loading data...')
data_path = "/home/ali/Desktop/data science project/data/"

train = pd.read_csv('train.csv',
                    dtype = {'msno': 'category',
                             'song_id': 'category',
                             'source_system_tab': 'category',
                             'source_screen_name': 'category',
                             'source_type': 'category'})

#labeling the msno to form the matrix.
le = LabelEncoder()
for col in ['msno']:
    print('Encoding {0}...'.format(col))
    train_vals = list(train[col].unique())
    le.fit(train_vals)
    train[col] = le.transform(train[col])
#adding cluster feature.
cluster = pd.read_csv(data_path + 'cluster_songs_20.csv',
                      dtype={'id': np.int8,
                             'song_id': 'category',
                             'cluster':'category'
                      })
cluster.drop('id', axis=1, inplace=True)
train = pd.merge(train, cluster, on='song_id', how='left')

#counting how many times a user listened to a specific cluster.
_list = train.groupby(['msno','cluster']).size().reset_index(name='count')
del train; del cluster;gc.collect()

matrix = np.zeros((len(_list['msno'].unique()), 20))

#filling the matrix user, cluster with counts.
for row in zip(_list['msno'], _list['cluster'],_list['count']):
    matrix[row[0]][int(row[1])] = row[2]

cols_a = ['msno','cluster_0','cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7','cluster_8','cluster_9',
          'cluster_10','cluster_11','cluster_12','cluster_13','cluster_14','cluster_15','cluster_16','cluster_17','cluster_18','cluster_19'
]
count_dict_a = dict([(key, []) for key in cols_a])
i = 0

#saving in a dict the user and 20 new scores as features and then writing to csv file.
for user_list in matrix:
    count_dict_a['msno'].append(le.inverse_transform(i))
    count_dict_a['cluster_0'].append(user_list[0])
    count_dict_a['cluster_1'].append(user_list[1])
    count_dict_a['cluster_2'].append(user_list[2])
    count_dict_a['cluster_3'].append(user_list[3])
    count_dict_a['cluster_4'].append(user_list[4])
    count_dict_a['cluster_5'].append(user_list[5])
    count_dict_a['cluster_6'].append(user_list[6])
    count_dict_a['cluster_7'].append(user_list[7])
    count_dict_a['cluster_8'].append(user_list[8])
    count_dict_a['cluster_9'].append(user_list[9])
    count_dict_a['cluster_10'].append(user_list[10])
    count_dict_a['cluster_11'].append(user_list[11])
    count_dict_a['cluster_12'].append(user_list[12])
    count_dict_a['cluster_13'].append(user_list[13])
    count_dict_a['cluster_14'].append(user_list[14])
    count_dict_a['cluster_15'].append(user_list[15])
    count_dict_a['cluster_16'].append(user_list[16])
    count_dict_a['cluster_17'].append(user_list[17])
    count_dict_a['cluster_18'].append(user_list[18])
    count_dict_a['cluster_19'].append(user_list[19])
    i += 1
df = pd.DataFrame(count_dict_a)
df = df[cols_a]
df.to_csv('kmeans_cluster_file.csv',index=False)


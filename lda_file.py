import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csc_matrix
import csv

data_path = '/home/ali/Desktop/data science project/data/'
#this file cluster the songs into topics(clusters) using lda. a new file lda_preprocess will convert to correct output csv file ready to merge with lightgbm model.

columns = ["msno","song_id","target"]
train_file = pd.read_csv(data_path + 'train.csv', skipinitialspace=True, usecols=columns)
print (train_file.head())

cols = ["msno","song_id"]
#labeling 
le = LabelEncoder()
for col in tqdm(cols):
    if train_file[col].dtype == "object":
        train_vals = list(train_file[col].unique())
        le.fit(train_vals)
        train_file[col] = le.transform(train_file[col])

print(train_file.head())

song_idx = list(train_file['song_id'].unique())
cnt = len(song_idx)
print(max(song_idx))

msno_nr = 30738
song_nr = 359500
#matrix = np.zeros((msno_nr, song_nr))


sys.exit(1)
print("starting filling the matrix ....")
for row in zip(train_file['msno'], train_file['song_id'], train_file['target']):
    matrix[row[0]][row[1]] = row[2]
    if row[1] > counter:
        counter = row[1]

del train_file; gc.collect();

matrix = csc_matrix(matrix)

print ("finished matrix complete ...")
lda = LatentDirichletAllocation(n_components=50, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(matrix)

filename = open("data/lda_distribution.csv", 'w')
fieldnames = ['song_id', 'topic_id', "distribution"]
writer = csv.DictWriter(filename, fieldnames=fieldnames)
writer.writeheader()
print("finished lda operation ....")
for topic_idx, topic in enumerate(lda.components_):
    message = "Topic #%d: " % topic_idx
    mess = ''
    for i in topic.argsort()[:-cnt - 1:-1]:
        writer.writerow({'song_id': le.inverse_transform(song_idx[i]), 'topic_id': topic_idx, 'distribution':topic[i]})


print()

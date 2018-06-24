from gensim import models
import gensim
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
pd.options.mode.chained_assignment = None
import gc
import csv


data_path = "/home/ali/Desktop/data science project/data/"


columns = ["msno","song_id","target"]
train_file = pd.read_csv(data_path + 'train.csv', skipinitialspace=True, usecols=columns)
print (train_file.head())

le = LabelEncoder()
train_vals = list(train_file['song_id'].unique())
le.fit(train_vals)
train_file['song_id'] = le.transform(train_file['song_id'])

#this file is the output of lda_file.py
lda = pd.read_csv(data_path + "lda_distribution_100.csv", skipinitialspace=True)



print(len(train_vals))
song_idx = list(lda['song_id'].unique())
cnt = len(song_idx)
del song_idx;gc.collect();
del train_file;del train_vals; gc.collect()
lst = []

for i in range(0,cnt):

    lst.append([])


counter = 0
for row in zip(lda['song_id'], lda['distribution']):
    song_id = row[0]
    dist = row[1]
    lst[song_id].append(dist)



print ('writing to file')
filename = open("lda_new100.csv", 'w')
fieldnames = ['song_id']
for i in range(0,100):
    fieldnames.append("topic" + str(i))
writer = csv.writer(filename)
writer.writerow(fieldnames)
for i in range (0, len(lst)):
    temp = []
    temp.append(le.inverse_transform(i))
    if lst[i]:
       for j in range(0, len(lst[i])):
           temp.append(lst[i][j])

    writer.writerow(temp)

    if i % 50000 == 0:
        print ("step: " + str(i))


print ('finished')
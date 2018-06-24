"""
Collaborative filtering w. matrix factorization using the alpenglow library.

Usage:
    python factorization.py path_to_csv_files

    test output is written to predictions_cf_mf.csv in the same folder

Todo:
    - Learn some pandas and clean this up.

Performance improves slightly if only items & users with sufficient entries 
are rated:
    ~10% items without ratings: 0.653
    items and users have 1+ ratings: 0.668
    items and users have 2+ ratings: 0.670
    items and users have 3+ ratings: 0.672
    items and users have 5+ ratings: 0.674
    items and users have 10+ ratings: 0.678
"""
import argparse
import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from alpenglow.offline.evaluation import NdcgScore, PrecisionScore, RecallScore
from alpenglow.offline.models import FactorModel
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Nr. of data points used for fitting the model, rest is used for validation.
n = 7200000

# Parse cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="folder where csv files are stored")
args = parser.parse_args()
path = os.path.join(args.path, "")
print("Path: " + path)

# Read data
print("Reading data..")
members = pd.read_csv(path + "members.csv", usecols=["msno"])
songs = pd.read_csv(path + "songs.csv", usecols=["song_id"])
test = pd.read_csv(path + "test.csv", usecols=["msno", "song_id"])
raw_data = pd.read_csv(path + "train.csv")

# Convert user and song ids to integers
print("Preprocessing..")
le_members = LabelEncoder()
le_songs = LabelEncoder()
le_members.fit(members["msno"].append(test["msno"]).append(raw_data["msno"]))
le_songs.fit(songs["song_id"].append(test["song_id"]).append(raw_data["song_id"]))  # Some song id's in test/train aren't in songs.csv
data = pd.DataFrame({
    "user" : le_members.transform(raw_data["msno"]),
    "item" : le_songs.transform(raw_data["song_id"]),
    "score": raw_data["target"].copy()
    })
test["user"] = le_members.transform(test["msno"])
test["item"] = le_songs.transform(test["song_id"])

del members; del songs; del raw_data; gc.collect()

# Train model
#for dimension, learning_rate, regularization_rate in product(
#        [10, 20, 50, 200],
#        [0.001, 0.01, 0.05, 0.1],
#        [0, 0.001, 0.01, 0.1, 1]):
dimension = 50
learning_rate = 0.01
regularization_rate = 0

print("Fitting model.. dim={}, learning_rate={}, regularization={}".format(
    dimension, learning_rate, regularization_rate))
train = data[:n].copy()
validate = data[n:].copy()
factor_model = FactorModel(
    dimension=dimension,
    learning_rate=learning_rate,
    regularization_rate=regularization_rate,
    seed=2,
    number_of_iterations=15,
)
factor_model.fit(train)

# Evaluate model
user_counts = train["user"].value_counts()
item_counts = train["item"].value_counts()
min_ratings = 10
def filter(i):
    item_id = validate["item"][i] 
    user_id = validate["user"][i] 
    return (item_counts.get(item_id, 0) >= min_ratings and
            user_counts.get(user_id, 0) >= min_ratings)
print("Evaluating model.. min_ratings={}".format(min_ratings))
val = validate.select(filter)
p = np.array(factor_model.predict(val))
p = (p - p.min()) / (p.max() - p.min())
auc = roc_auc_score(np.array(val["score"]), p)
print("AUC: {:.3f}".format(auc))

# Test
print("Test set predictions..")
p = np.array(factor_model.predict(test))
p = (p - p.min()) / (p.max() - p.min())
res = pd.DataFrame()
res["id"] = range(len(p))
res["target"] = p
res["target"] = res["target"].clip(0, 1)
res.to_csv(path + "predictions_cf_mf.csv", index=False)
print("Done, results are in predictions_cf_mf.csv")

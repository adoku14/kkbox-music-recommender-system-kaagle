""" 
Combines the results of the lgbm, clustering, matrix factorization and
logistic regression models. Looks for predictions_lgbm.csv,
predictions_cluster.csv predictions_cf_mf.csv and
predictions_logistic_regression.csv files in the given path.

Either run the corresponding scripts to generate those predictions first, or
download them from the studento server.

Usage:
    python hybrid.py path_to_csv_files

    test output is written to the same folder
"""
import argparse
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Parse cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="folder where csv files are stored")
args = parser.parse_args()
path = os.path.join(args.path, "")
print("Path: " + path)
#path = "/home/oliver/tmp/datasets/kaggle/"

# Read train data
print("Reading training data..")
test  = pd.read_csv(path + "test.csv" , usecols=["msno", "song_id"])
train = pd.read_csv(path + "train.csv", usecols=["msno", "song_id", "target"])
le_members = LabelEncoder()
le_songs = LabelEncoder()
le_members.fit(train["msno"].append(test["msno"]))
le_songs.fit(train["song_id"].append(test["song_id"]))
train = pd.DataFrame({
    "user"  : le_members.transform(train["msno"]),
    "song"  : le_songs.transform(train["song_id"]),
    "target": train["target"].copy()
    })
test = pd.DataFrame({
    "user"  : le_members.transform(test["msno"]),
    "song"  : le_songs.transform(test["song_id"])
    })

# Count user and song entries
print("Calculating user & song counts..")
user_counts = train[["user", "target"]].groupby("user").count();
song_counts = train[["song", "target"]].groupby("song").count(); 
user_counts.columns = ["user_count"]
song_counts.columns = ["song_count"]
train = pd.merge(train, user_counts, how="left", left_on="user", right_index=True)
train = pd.merge(train, song_counts, how="left", left_on="song", right_index=True)
test  = pd.merge(test , user_counts, how="left", left_on="user", right_index=True)
test  = pd.merge(test , song_counts, how="left", left_on="song", right_index=True)
test["user_count"].fillna(0, inplace=True)
test["song_count"].fillna(0, inplace=True)

# Read predictions made by the lgbm and matrix factorization models.
print("Reading csv files..")
baseline = pd.read_csv(path + "predictions_lgbm.csv")
cluster = pd.read_csv(path + "predictions_cluster.csv")
mf = pd.read_csv(path + "predictions_cf_mf.csv")
lr = pd.read_csv(path + "predictions_logistic_regression.csv")

# Combine predictions. First, clustering and matrix factorization are combined
# using fixed weights. The result of that is combined with lgbm, giving more
# weight to lgbm for users/songs where there is insufficient training data for
# collaborative filtering (matrix factorization) to make meaningful
# predictions. Finally, those results are merged with the logistic regression
# model. Sorry if this is hairy.
print("Writing output csv..")
w_cl    = 0.60  # Weight given to cluster predictions
w_lgbm  = 0.50  # Minimum weight given to lgbm predictions
w_lr    = 0.15  # Weight of logistic regression
target  = w_cl * cluster["target"] + (1 - w_cl) * mf["target"]
weights = (np.clip(np.log2(1 + np.sqrt(test["user_count"] * test["song_count"])), 0, 10) / 10) * (1 - w_lgbm)
target  = weights * target + (1 - weights) * baseline["target"]
weights = np.repeat(w_lr, len(test))
weights[:10000] = 0; weights[-10000:] = 0;  # No density information at start/end of data
target = (1 - weights) * target + weights * lr["target"]
df = pd.DataFrame()
df["id"] = range(len(target))
df["target"] = target
df.to_csv(path + "baseline_cluster_mf_lr.csv", index=False)



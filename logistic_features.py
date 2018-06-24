"""
Script for generating features needed for the logistic regression model. 

This takes several hours to run. If you just want to run the logistic
regression model, download the precomputed lr_features_{train,test}.csv
files from the studento server.

Usage:
    python logistic_features.py path_to_csv_files

    feature csv files are written to lr_features_{train,test}.csv
"""

import argparse
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder


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

# User and song average ratings
print("Calculating user & song averages..")
user_means = train[["user", "target"]].groupby("user").mean();
song_means = train[["song", "target"]].groupby("song").mean(); 
user_means.columns = ["user_mean"]
song_means.columns = ["song_mean"]
train = pd.merge(train, user_means, how="left", left_on="user", right_index=True)
train = pd.merge(train, song_means, how="left", left_on="song", right_index=True)
test  = pd.merge(test , user_means, how="left", left_on="user", right_index=True)
test  = pd.merge(test , song_means, how="left", left_on="song", right_index=True)
test["user_mean"].fillna(0.5, inplace=True)
test["song_mean"].fillna(0.5, inplace=True)

# User and song occurrences
print("Calculating user & song counts..")
user_pct = train[["user", "target"]].groupby("user").count() / len(train);
song_pct = train[["song", "target"]].groupby("song").count() / len(train); 
user_pct.columns = ["user_pct"]
song_pct.columns = ["song_pct"]
train = pd.merge(train, user_pct, how="left", left_on="user", right_index=True)
train = pd.merge(train, song_pct, how="left", left_on="song", right_index=True)
user_pct = test[["user", "song"]].groupby("user").count() / len(test);
song_pct = test[["song", "user"]].groupby("song").count() / len(test); 
user_pct.columns = ["user_pct"]
song_pct.columns = ["song_pct"]
test  = pd.merge(test, user_pct, how="left", left_on="user", right_index=True)
test  = pd.merge(test, song_pct, how="left", left_on="song", right_index=True)

# Density features
print("Calculating densities for train and test..")
for df in [train, test]:
    print("Calculating densities in 100k window..")
    w = 100000 * 2 + 1
    def density(a): return (np.sum(a == a[w//2]) - 1) / (w - 1)
    df["item_density_100k"] = df["song"].rolling(window=w, center=True).apply(density)
    df["item_density_100k"].fillna(0, inplace=True)
    df["item_density_relative_100k"] = df["item_density_100k"] / df["song_pct"]
    df["user_density_100k"] = df["user"].rolling(window=w, center=True).apply(density)
    df["user_density_100k"].fillna(0, inplace=True)
    df["user_density_relative_100k"] = df["user_density_100k"] / df["user_pct"]

    print("Calculating densities in 50k window..")
    w = 50000 * 2 + 1
    df["item_density_50k"] = df["song"].rolling(window=w, center=True).apply(density)
    df["item_density_50k"].fillna(0, inplace=True)
    df["user_density_50k"] = df["user"].rolling(window=w, center=True).apply(density)
    df["user_density_50k"].fillna(0, inplace=True)

    print("Calculating densities in 10k window..")
    w = 10000 * 2 + 1
    df["item_density_10k"] = df["song"].rolling(window=w, center=True).apply(density)
    df["item_density_10k"].fillna(0, inplace=True)
    df["item_density_relative_10k"] = df["item_density_10k"] / df["song_pct"]
    df["user_density_10k"] = df["user"].rolling(window=w, center=True).apply(density)
    df["user_density_10k"].fillna(0, inplace=True)
    df["user_density_relative_10k"] = df["user_density_10k"] / df["user_pct"]

    print("Calculating densities in 1k window..")
    w = 1000 * 2 + 1
    df["item_density_1k"] = df["song"].rolling(window=w, center=True).apply(density)
    df["item_density_1k"].fillna(0, inplace=True)
    df["item_density_relative_1k"] = df["item_density_1k"] / df["song_pct"]
    df["user_density_1k"] = df["user"].rolling(window=w, center=True).apply(density)
    df["user_density_1k"].fillna(0, inplace=True)
    df["user_density_relative_1k"] = df["user_density_1k"] / df["user_pct"]

# Omitting some features that weren't helpful on validation set to reduce file size
print("Saving features to csv files..")
features = ["user_density_1k", "item_density_1k", "user_density_10k", 
            "item_density_10k", "user_density_50k", "item_density_50k", 
            "user_density_100k", "item_density_100k"]
train[features + ["target"]].to_csv(path + "lr_features_train.csv", index=False)
test [features].to_csv(path + "lr_features_test.csv" , index=False)


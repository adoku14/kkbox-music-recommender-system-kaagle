"""
A logistic regression model w. features based on sequence information.

Requires the lr_features_{train,test}.csv files, which can either be found 
on studento or calculated using logistic_features.py

Usage:
    python logistic.py path_to_csv_files

    test output is written to predictions_logistic_regression.csv in 
    the same folder
"""

import argparse
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn import linear_model


# Parse cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="folder where csv files are stored")
args = parser.parse_args()
path = os.path.join(args.path, "")
print("Path: " + path)
#path = "/home/oliver/tmp/datasets/kaggle/"

# Read feature files
print("Reading data..")
train = pd.read_csv(path + "lr_features_train.csv")
test  = pd.read_csv(path + "lr_features_test.csv")

# Select features. 10k windows do best on validation, both by themselves and 
# when combined with user and song means.
features = ["user_density_1k", "item_density_1k", "user_density_10k", 
            "item_density_10k", "user_density_50k", "item_density_50k", 
            "user_density_100k", "item_density_100k", 
            ]
features = ["user_density_10k", "item_density_10k"]
X = train[features].as_matrix()
y = np.array(train["target"])
X_test = test[features].as_matrix()

#V = len(train) // 500
#X_train, y_train = X[:-V], y[:-V]
#X_valid, y_valid = X[-V:], y[-V:]
X_train, y_train = X, y

print("Training model..")
logreg = linear_model.LogisticRegression(C=1e5, random_state=2)
logreg.fit(X_train, y_train)

# Validation
#Z = logreg.predict_proba(X_valid)[:,1]
#score = roc_auc_score(y_valid, Z)
#print("Validation auc: " + str(score))

print("Test predictions..")
target = logreg.predict_proba(X_test)[:,1]
df = pd.DataFrame()
df["id"] = range(len(target))
df["target"] = target
df.to_csv(path + "predictions_logistic_regression.csv", index=False)


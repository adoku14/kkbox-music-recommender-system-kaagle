# Info 
This is a group project for a Master Course in University of Antwerp called Data Science Project. We were registered in a online competition in kaggle website named kkbox Music Recommendation System. The final Ranking of our group is 42 out of 1081 teams. My contribution in this project are:

- LGBM model and feature Engineering
- Using clustering for creating new features which importantly improved the score

We have used a lot of strategies but we ended up with the best models as discribed below

# Generating the Predictions

Our final submission combines predictions from four independent models:

- LGBM
- Matrix factorization
- Logistic regression
- Clustering

To make it easy to tweak each model independently, each has its own script
that outputs a csv file with predictions. These scripts are
lightGBM_genreCount_0_6940.py, cf-mf/factorization.py and logistic.py. The
script for the clustering model are kmeans_cluster.py, outputing the csv file with clusters. Afterwards, run score_msno_to_cluster.py to get a score for each user in each cluster. 

Once you have the prediction files for the four models, run hybrid.py to
combine the predictions and output the final csv file.

# Dependencies

The SciPy stack, sklearn, tqdm, lightgbm and Alpenglow are needed to generate
the submission completely from scratch.

To just run hybrid.py, only numpy, pandas and sklearn are needed.

# Generating the Predictions

Our final submission combines predictions from four independent models:

- LGBM
- Matrix factorization
- Logistic regression
- Clustering

To make it easy to tweak each model independently, each has its own script
that outputs a csv file with predictions. These scripts are
lightGBM_genreCount_0_6940.py, cf-mf/factorization.py and logistic.py. The
script for the clustering model went missing, its predictions (and those of
the other models) can be found on studento.

Once you have the prediction files for the four models, run hybrid.py to
combine the predictions and output the final csv file.

# Dependencies

The SciPy stack, sklearn, tqdm, lightgbm and Alpenglow are needed to generate
the submission completely from scratch.

To just run hybrid.py, only numpy, pandas and sklearn are needed.

import fastbook
from fastbook import *
from fastai.collab import *

#This script trains a collaborative filtering learner and predicts a score for a user, item pair
#Requirements: csv data with user_id, item_id, and rating columns

#Replace * with name of csv file
ratings = pd.read_csv('*.csv')  
dls = CollabDataLoaders.from_df(ratings, valid_pct=0.2, user_name='user_id', item_name='item_id', rating_name='rating', bs=64)

#Create matrix factorization learner with 20 factors
#y_range is the rating range, set to 1, 5.0 for 1-5 star rating
learn = collab_learner(dls, n_factors=20, y_range=(0, 1.0))
learn.fit_one_cycle(5, 2e-2, wd=0.1)

#Need to create a copy to get predictions because of a small FastAI bug:
new_df = ratings.copy()
dl = learn.dls.test_dl(new_df)

#Print a sample of the input, output, and ground truth:
inp, preds, targets = learn.get_preds(with_input=True, dl=dl)
print("       First 15 inputs\n      (user_id, item_id):\n", inp[0:15])
print("First 15 predictions:\n", preds[0:15])
print("First 15 actual ratings (targets):\n", targets[0:15])

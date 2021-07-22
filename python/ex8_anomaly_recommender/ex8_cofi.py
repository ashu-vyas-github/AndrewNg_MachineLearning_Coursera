# Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy

import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy import optimize

import ano_rec_funcs

# =============== Part 1: Loading movie ratings dataset ================
print("Loading movie ratings dataset.\n\n")

# Load data
data = loadmat('./ex8_movies.mat')
Y, R = data['Y'], data['R']

print("Average rating for movie 1 (Toy Story): %f / 5" % numpy.mean(Y[0, R[0, :]]))

# We can "visualize" the ratings matrix by plotting it with imshow
plt.figure(figsize=(8, 8))
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()

# ============ Part 2: Collaborative Filtering Cost Function ===========
#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data = loadmat('./ex8_movieParams.mat')
X, Theta = data['X'], data['Theta']
num_users, num_movies, num_features = data['num_users'], data['num_movies'], data['num_features']

#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, 0:num_users]
R = R[:num_movies, 0:num_users]

#  Evaluate cost function
J, _ = ano_rec_funcs.cofi_cost_func(numpy.concatenate(
    [X.ravel(), Theta.ravel()]), Y, R, num_users, num_movies, num_features)

print("Computed cost J:", J)
print("Expected cost J: 22.22")

# ============== Part 3: Collaborative Filtering Gradient ==============
print("\nChecking Gradients (without regularization)...\n")

#  Check gradients by running checkcostFunction
ano_rec_funcs.check_cost_function(ano_rec_funcs.cofi_cost_func)

# ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Evaluate cost function
J, _ = ano_rec_funcs.cofi_cost_func(numpy.concatenate(
    [X.ravel(), Theta.ravel()]), Y, R, num_users, num_movies, num_features, 1.5)

print("\nComputed cost J at loaded parameters (lambda = 1.5): %.2f" % J)
print("Expected cost J at loaded parameters (lambda = 1.5): 31.34")

# ======= Part 5: Collaborative Filtering Gradient Regularization ======
print("\nChecking Gradients (with regularization)...\n")

#  Check gradients by running checkcostFunction
ano_rec_funcs.check_cost_function(ano_rec_funcs.cofi_cost_func, 1.5)

# ============== Part 6: Entering ratings for a new user ===============
print("\n")
movie_list = ano_rec_funcs.load_movie_list()
num_movies = len(movie_list)

my_ratings = numpy.zeros(num_movies)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
# Note that the index here is ID-1, since we start index from 0.
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print("New user ratings:")
print("-----------------")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated %d stars: %s" % (my_ratings[i], movie_list[i]))

# ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users

#  Load data
data = loadmat('./ex8_movies.mat')
Y, R = data['Y'], data['R']

#  Add our own ratings to the data matrix
Y = numpy.hstack([my_ratings[:, None], Y])
R = numpy.hstack([(my_ratings > 0)[:, None], R])

#  Normalize Ratings
Ynorm, Ymean = ano_rec_funcs.normalize_ratings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 10

# Set Initial Parameters (Theta, X)
X = numpy.random.randn(num_movies, num_features)
Theta = numpy.random.randn(num_users, num_features)
initial_parameters = numpy.concatenate([X.ravel(), Theta.ravel()])

# Set options for scipy.optimize.minimize
options = {'maxiter': 100}

# Set Regularization
lambda_ = 10
res = optimize.minimize(lambda x: ano_rec_funcs.cofi_cost_func(x, Ynorm, R, num_users, num_movies,
                                                               num_features, lambda_), initial_parameters, method='TNC', jac=True, options=options)
theta = res.x

# Unfold the returned theta back into U and W
X = theta[:num_movies*num_features].reshape(num_movies, num_features)
Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

print("\n\nRecommender system learning completed.\n")

# # ================== Part 8: Recommendation for you ====================
pred = numpy.dot(X, Theta.T)
my_predictions = pred[:, 0] + Ymean
movie_list = ano_rec_funcs.load_movie_list()

ix = numpy.argsort(my_predictions)[::-1]

print("Top recommendations for you:")
print('----------------------------')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % (my_predictions[j], movie_list[j]))

print('\nOriginal ratings provided:')
print('--------------------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movie_list[i]))

print("\n\n\nDone\n\n\n")

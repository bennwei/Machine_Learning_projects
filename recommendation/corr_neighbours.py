# Charpter 8 recommendation from Building Machine Learning Systems With Python
# building a recommendation systems based on customer movie ratings using Movielens ML-100k data.
# A nearest neighborhood approach to recommendations: to know how a user will rate a movie, find the users
# most similar to them, and look at their ratings.
# Eric Liao @2017

from __future__ import print_function
import numpy as np
from load_ml100k import get_train_test
from scipy.spatial import distance
from sklearn import metrics

from normalization import Normalize_Positive


def predict(train_data, plot_matrix=False):

    binary = (train_data > 0)
    norm = Normalize_Positive(axis=1)
    train = norm.fit_transform(train_data)
    # visualize the values of the binary matrix as an image
    if plot_matrix:
        from matplotlib import pyplot as plt
        # plot just 200x200 area
        plt.imshow(binary[:200, :200], interpolation='nearest')
        plt.savefig('binary_figure.png')

    # compute all pair-wise distances of all user ratings
    dists = distance.pdist(binary, 'correlation')
    # converts between condensed distance matrices and square distance matrices
    dists = distance.squareform(dists)

    neighbors = dists.argsort(axis=1)  # sort along rows-wise, return index
    filled = train.copy()

    # The rating prediction algorithm will be (in pseudo code) as follows:
    # 1. For each user, rank every other user in terms of closeness. For this step,
    # we will use the binary matrix and use correlation as the measure of
    # closeness (interpreting the binary matrix as zeros and ones allows
    # us to perform this computation).
    # 2. When we need to estimate a rating for a (user, movie) pair, we look at all the
    # users who have rated that movie and split them into two: the most similar
    # half and the most dissimilar half. We then use the average of the most similar
    # half as the prediction.

    for user in range(filled.shape[0]):
        neighbor_user = neighbors[user, 1:]
        for movie in range(filled.shape[1]):
            # get relevant reviews in order!
            revs = [train[neigh, movie] for neigh in neighbor_user if binary[neigh, movie]]

            if len(revs):
                n = len(revs)
                n //= 2
                n += 1
                revs = revs[:n]
                filled[user, movie] = np.mean(revs)

    return norm.inverse_transform(filled)


def main(transpose_inputs=False):
    train, test = get_train_test(random_state=12)
    if transpose_inputs:
        train = train.T
        test = test.T

    predicted = predict(train, plot_matrix=True)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 score (binary {} neighbours): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))

if __name__ == '__main__':
    main()
    main(transpose_inputs=True)



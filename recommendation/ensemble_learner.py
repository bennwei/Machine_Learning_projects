# Charpter 8 recommendation from Building Machine Learning Systems With Python
# building a recommendation systems based on customer movie ratings using Movielens ML-100k data.
# A stacked leaner (ensemble_learning) combining several predictors for predicting recommendations
# Eric Liao @2017

import numpy as np
import load_ml100k
import regression
import corr_neighbours
from sklearn import linear_model, metrics
import normalization


def stacked_predict(train_data):
    # Stacked prediction: when fitting hyperparameters, though, we need two layers of training/testing splits: a first, higher-level split,
    # and then, inside the training split, a second split to be able to fit the stacked learner.

    tr_train, tr_test = load_ml100k.get_train_test(train_data, random_state=34)

    # Call all the methods we previously defined:
    # these have been implemented as functions:
    tr_prediction_0 = regression.predict(tr_train)
    tr_prediction_1 = regression.predict(tr_train.T).T
    tr_prediction_2 = corr_neighbours.predict(tr_train)
    tr_prediction_3 = corr_neighbours.predict(tr_train.T).T
    tr_prediction_4 = normalization.predict(tr_train)
    tr_prediction_5 = normalization.predict(tr_train.T).T

    # Now assemble these predictions into a single array
    stacked_learner = np.array([
        tr_prediction_0[tr_test > 0],
        tr_prediction_1[tr_test > 0],
        tr_prediction_2[tr_test > 0],
        tr_prediction_3[tr_test > 0],
        tr_prediction_4[tr_test > 0],
        tr_prediction_5[tr_test > 0],
    ]).T

    # Fit a simple linear regression
    linear_leaner = linear_model.LinearRegression()
    linear_leaner.fit(stacked_learner, tr_test[tr_test > 0])

    # apply the whole process to the testing split and evaluate
    stacked_te = np.array([
        tr_prediction_0.ravel(),
        tr_prediction_1.ravel(),
        tr_prediction_2.ravel(),
        tr_prediction_3.ravel(),
        tr_prediction_4.ravel(),
        tr_prediction_5.ravel(),
    ]).T

    return linear_leaner.predict(stacked_te).reshape(tr_train.shape)

def average_predict(train):
    # Averaging of predictions
    predicted0 = regression.predict(train)
    predicted1 = regression.predict(train.T).T
    predicted2 = corr_neighbours.predict(train)
    predicted3 = corr_neighbours.predict(train.T).T
    predicted4 = normalization.predict(train)
    predicted5 = normalization.predict(train.T).T
    stack = np.array([
        predicted0,
        predicted1,
        predicted2,
        predicted3,
        predicted4,
        predicted5,
        ])
    return stack.mean(0)


def main():
    train,test = load_ml100k.get_train_test(random_state=12)
    predicted = stacked_predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('Results from ensemble_learner.py stacked prediction')
    print('R2 stacked: {:.2%}'.format(r2))

if __name__ == '__main__':
    main()


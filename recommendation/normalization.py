# Charpter 8 recommendation from Building Machine Learning Systems With Python
# building a recommendation systems based on customer movie ratings using Movielens ML-100k data.
# normalize the rating data to remove obvious movie or user-specific effects: conversion to z-scores.
# Eric Liao @2017

import numpy as np

class Normalize_Positive(object):

    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, features, y=None):
        # compute the mean and standard deviation of the values that are not zero.

        if self.axis == 1:
            features = features.T
        binary = (features > 0)
        count = binary.sum(axis=0)

        # to avoid division by zero, set zero counts to one:
        count[count == 0] = 1

        self.mean = features.sum(axis=0) / count

        # Compute variance by average squared difference to the mean, but only
        # consider differences where binary is True (i.e., where there was a
        # true rating):

        diff = (features - self.mean) * binary
        diff **= 2

        # regularize the estimate of std by adding 0.1
        self.std = np.sqrt(0.1 + diff.sum(axis=0) / count)

        return self

    def transform(self, features):
        # transform method needs to take care of maintaining the binary structure
        if self.axis == 1:
            features = features.T
        binary = (features >0)
        features = features - self.mean
        features /= self.std
        features *= binary
        if self.axis ==1:
            features = features.T
        return features

    def inverse_transform(self, features, copy=True):
        # performs the inverse operation to transform, so that the return value has the same shape as the input.
        if copy:
            features = features.copy()

        if self.axis == 1:
            features = features.T

        features *= self.std
        features += self.mean
        if self.axis == 1:
            features = features.T
        return features

    def fit_transform(self, features):
        # combines both the fit and transform operations.
        return self.fit(features).transform(features)

def predict(train):
        normalizer = Normalize_Positive()
        train = normalizer.fit_transform(train)
        return normalizer.inverse_transform(train * 0.)

def main(transpose_inputs=False):
        from load_ml100k import get_train_test
        from sklearn import metrics
        train, test = get_train_test(random_state=12)
        if transpose_inputs:
            train = train.T
            test = test.T
        predicted = predict(train)
        r2_score = metrics.r2_score(test[test > 0], predicted[test > 0])

        print('R2 score ({} normalization): {:.1%}'.format(('movie' if transpose_inputs else 'user'), r2_score))

if __name__ == '__main__':
    main()
    main(transpose_inputs=True)











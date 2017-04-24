# Charpter 8 recommendation from Building Machine Learning Systems With Python
# building a recommendation systems based on customer movie ratings using Movielens ML-100k data.
# A regression approach to recommendations
# Eric Liao @2017

import numpy as np
from sklearn.linear_model import ElasticNetCV
from normalization import Normalize_Positive
from sklearn import metrics


def predict(train_data):
    binary = (train_data > 0)
    model = ElasticNetCV(fit_intercept=True, alphas=[0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])
    norm = Normalize_Positive()
    train_data = norm.fit_transform(train_data)

    filled = train_data.copy()
    # we iterate over all the users, and each time learn a regression model based only on the data
    for u in range(train_data.shape[0]):
        # remove the current user for training
        cur_train = np.delete(train_data, u, axis=0)
        # binary records whether this rating is present
        bu = binary[u]
        # fit the current user based on everybody else
        if np.sum(bu) > 5:
            model.fit(cur_train[:, bu].T, train_data[u, bu])
            # Fill the values that were not there already
            filled[u, ~bu] = model.predict(cur_train[:, ~bu].T)

    return norm.inverse_transform(filled)


def main(transpose_inputs=False):
    from load_ml100k import get_train_test
    train,test = get_train_test(random_state=12)
    if transpose_inputs:
        train = train.T
        test = test.T
    filled = predict(train)
    r2 = metrics.r2_score(test[test > 0], filled[test > 0])

    print('R2 score ({} regression): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))

if __name__ == '__main__':
    main()
    main(transpose_inputs=True)




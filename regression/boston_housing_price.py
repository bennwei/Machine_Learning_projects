# Charpter 7 regression from Building Machine Learning Systems With Python
# Use a simple and penalized regression to predict boston housing price and evaluate generalization error
# Eric Liao @2017

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

boston = load_boston()
x = boston.data
y = boston.target

# Fit a regression model
lr = LinearRegression()
lr.fit(x, y)

# compute rmse, `residues_` contains the sum of the squared residues
rmse = np.sqrt(lr.residues_ / len(x))
print('RMSE from ordinary least squares (OLS) regression: {}'.format(rmse))

# Plot figures
fig, ax = plt.subplots()
# Plot a diagonal for reference
ax.plot([0, 50], [0, 50], '-', color=(.9,.3,.3), lw=4)

# Plot prediction vs. real values
ax.scatter(lr.predict(x), boston.target)

ax.set_xlabel('predicted')
ax.set_ylabel('real')
fig.savefig('Figure_07_08.png')

# Then try to fit several forms of penalized regression
for name, met in [('linear regression', LinearRegression()),
                  ('lasso', Lasso()),
                  ('elastic-net(0.5)', ElasticNet(alpha=0.5)),
                  ('lasso(0.5)', Lasso(alpha=0.5)),
                  ('ridge(0.5)', Ridge(alpha=0.5)),]:

    met.fit(x, y)

    # Predict the whole data
    pred = met.predict(x)
    r2_train = r2_score(y, pred)

    # Use 10-fold cross-validation to estimate generalization error
    kf = KFold(n_splits=10)
    pred = np.zeros_like(y)
    for train, test in kf.split(x):
        # print("TRAIN:", train, "TEST:", test)
        met.fit(x[train], y[train])
        pred[test] = met.predict(x[test])

    r2_cv = r2_score(y, pred)

    # Print out results
    print('Method: {}'.format(name))
    print('R^2 on training: {}'.format(r2_train))
    print('R^2 on 10-fold CV: {}'.format(r2_cv))

    print()







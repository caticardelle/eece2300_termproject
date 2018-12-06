import regression
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

y = regression.y.to_frame()
x = regression.x.ix[:, regression.lassocv(regression.x, y)]

y['CrimeCategory'] = np.where(y['ViolentCrimesPerPop']<.25,'low','medium')
y['CrimeCategory'] = np.where(y['ViolentCrimesPerPop']>.4,'high', y['CrimeCategory'])
Y = y['CrimeCategory']


def decisiontree(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47, test_size=0.25)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
    print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
    clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)
    clf.fit(X_train, y_train)
    print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
    print('Accuracy Score on the test data: ', accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
    dtc = DecisionTreeClassifier()
    cv_scores = cross_val_score(dtc, X, y, cv=10)
    sns.distplot(cv_scores)
    plt.title('Average score: {}'.format(np.mean(cv_scores)))
    plt.show()

    parameter_grid = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [None, 1, 2, 3, 4, 5],
                      'max_features': [None, 1, 2, 3, 4]}

    grid_search = GridSearchCV(dtc, param_grid=parameter_grid, cv=10)

    grid_search.fit(X, y)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    dtc = grid_search.best_estimator_
    print(dtc)


decisiontree(x, Y)

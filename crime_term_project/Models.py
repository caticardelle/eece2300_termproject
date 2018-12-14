import regression
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

y = regression.y.to_frame()
x = regression.x.ix[:, regression.lassocv(regression.x, y)]
y['CrimeCategory'] = np.where(y['ViolentCrimesPerPop']<.25,'low','medium')
y['CrimeCategory'] = np.where(y['ViolentCrimesPerPop']>.4,'high', y['CrimeCategory'])
Y = y['CrimeCategory']

x.to_csv('C:\\Users\\catic\\Documents\\EECE 2300\\python\\crime_term_project\\data\\preprocessed\\x.csv')
y.to_csv('C:\\Users\\catic\\Documents\\EECE 2300\\python\\crime_term_project\\data\\preprocessed\\y.csv')


def decision_tree(X, y):

    dtc = DecisionTreeClassifier(random_state= 1)
    cv_scores = cross_val_score(dtc, X, y, cv=10)
    sns.distplot(cv_scores)
    plt.title('Average score: {}'.format(np.mean(cv_scores)))
    plt.show()

    parameter_grid = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [None, 1, 2, 3, 4, 5],
                      'max_features': [None, 1, 2, 3, 4, 5],
                      'max_leaf_nodes': [None, 2, 3, 4, 5]}
    grid_search = GridSearchCV(dtc, param_grid=parameter_grid, cv=10)

    grid_search.fit(X, y)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    dtc = grid_search.best_estimator_
    predictions = dtc.predict(X)
    print(accuracy_score(y, predictions))
    print(confusion_matrix(y, predictions, labels =['low', 'medium', 'high']))
    print(dtc)



#decision_tree(x, Y)
#randforrest(x, Y)

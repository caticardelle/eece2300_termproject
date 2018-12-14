import data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn import metrics
from sklearn import preprocessing

df = data.load_data('C:\\Users\\catic\\Documents\\EECE 2300\\python\\crime_term_project\\data\\raw\\communities.data.txt')
df2 = data.summarize_data(df)
df_attributes = data.label_data('C:\\Users\\catic\\Documents\\EECE 2300\\python\\crime_term_project\\data\\raw\\communities.attributes.txt', df2)
cleaned_df = data.clean_data(df_attributes)
cleaned_df = cleaned_df.replace('?', np.NaN)
cleaned_df = cleaned_df.dropna(axis=0)


x = cleaned_df.drop(['communityname','ViolentCrimesPerPop'], axis = 1)
y = cleaned_df['ViolentCrimesPerPop']
x_labels = x.columns
x_as_array = x.values
min_max_scale = preprocessing.MinMaxScaler()
x_scaled = min_max_scale.fit_transform(x_as_array)
x = pd.DataFrame(x_scaled)
x.columns = x_labels

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=.3, random_state=1)

def linreg(x,y):
    """
    FUnction for linear regression
    :param x: Attributes
    :param y: target variable
    :return: MSE
    """
    lm = LinearRegression()
    lm.fit(train_x, train_y)
    predict_test_lm = lm.predict(test_x)
    plt.figure(figsize=(15, 10))
    mse = metrics.mean_squared_error(test_y, predict_test_lm)
    ft_importances_lm = pd.Series(lm.coef_, index=x.columns)
    sorted_coefs = ft_importances_lm.sort_values()
    print (sorted_coefs)
    sorted_coefs.plot(kind='barh')
    plt.title("Linear Regression Coefficients \n Mean Squared Error = %f" % mse, fontsize=18)
    plt.xlim(-.6, .6)
    plt.show()
    print(mse)


def lassocv(x, y):
    """
    Function for Lasso cross validation
    :param x: Attributes
    :param y: Target
    :return: MSE and list of attributes with highest weight
    """
    reg = LassoCV(cv=10, random_state=1).fit(x, y)
    predictions = reg.predict(x)
    mse = metrics.mean_squared_error(y, predictions)
    plt.figure(figsize=(15, 10))
    coefs = pd.Series(reg.coef_, index=x.columns)
    sorted_coefs = coefs.sort_values()
    top_coefs = abs(sorted_coefs).sort_values(ascending=False).head(80)
    sorted_coefs.plot(kind='barh')
    plt.title("Lasso Coefficents \n Mean Squared Error = %f" % mse, fontsize=18)
    plt.xlim(-.6, .6)
    plt.show()
    print(reg.alpha_)
    print(mse)
    return list(top_coefs.index)


def ridgecv(x, y):
    """
    Function for Ridge with Cross Validation
    :param x: Attributes
    :param y: Target
    :return: MSE
    """
    reg = RidgeCV(cv=10).fit(x, y)
    predictions = reg.predict(x)
    mse = metrics.mean_squared_error(y, predictions)
    plt.figure(figsize=(15, 10))
    ft_importances_lm = pd.Series(reg.coef_[0], index=x.columns).sort_values()
    absolute_coefs = pd.Series(reg.coef_[0], index=x.columns)
    print(absolute_coefs.sort_values(ascending=False))
    ft_importances_lm.plot(kind='barh')
    plt.title("Ridge Coefficents \n Mean Squared Error = %f" % mse, fontsize=18)
    plt.xlim(-.6, .6)
    plt.show()
    print(reg.alpha_)
    return mse


def elasticnetcv(x, y):
    """
    FUnction for Elastic Net with Cross Validation
    :param x: Attributes
    :param y: Target
    :return: MSE
    """
    reg = ElasticNetCV(cv=10, random_state=1).fit(x, y)
    predictions = reg.predict(x)  # predicts the target variable
    mse = metrics.mean_squared_error(y, predictions)  # gets MSE from the true values and the predicted values
    plt.figure(figsize=(15, 10))
    ft_importances_lm = pd.Series(reg.coef_, index=x.columns)
    sorted_coefs = ft_importances_lm.sort_values()
    print(sorted_coefs.sort_values(ascending=False))
    sorted_coefs.plot(kind='barh')
    plt.title("Elastic Net Coefficents \n Mean Squared Error = %f" % mse, fontsize=18)
    plt.xlim(-.6, .6)
    plt.show()
    print(reg.alpha_)
    return mse


def visualization(topcoefs):
    """
    Shows correlation between violent crimes and highest ranked attributes by lasso

    :param topcoefs:
    :return:
    """
    plt.subplot(221)
    plt.scatter(cleaned_df.ix[:, topcoefs[0]], cleaned_df.ix[:, 'ViolentCrimesPerPop'], color='red')
    plt.title(topcoefs[0])
    plt.subplot(222)
    plt.scatter(cleaned_df.ix[:, topcoefs[1]], cleaned_df.ix[:, 'ViolentCrimesPerPop'], color='red')
    plt.title(topcoefs[1])
    plt.subplot(223)
    plt.scatter(cleaned_df.ix[:, topcoefs[2]], cleaned_df.ix[:, 'ViolentCrimesPerPop'], color='red')
    plt.title(topcoefs[2])
    plt.subplot(224)
    plt.scatter(cleaned_df.ix[:, topcoefs[3]], cleaned_df.ix[:, 'ViolentCrimesPerPop'], color='red')
    plt.title(topcoefs[3])
    plt.show()



fig, ax = plt.subplots()
size = cleaned_df['racepctblack']*100
cleaned_df.plot.scatter(x='PctIlleg', y='PctKids2Par', s=size, c='ViolentCrimesPerPop',
                        colormap='Reds', ax=ax)
plt.ylabel("Pct kids in family housing w/ two parents")
plt.xlabel("Pct kids born to never married")


#print(linreg(x,y))
#print(lassocv(x,y))
print(visualization(lassocv(x,y)))
#print(ridgecv(x,y))
#print(elasticnetcv(x,y))

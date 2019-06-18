"""
# House Price Prediction

### Project Information:
#### Kaggle competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/
#### Author: Yifei (Faye) Liu
#### Date: Sep 2018

1. **Understand the goal**. The goal for this project is predicting the house price which is a continuous variable. 
2. **Understand the inputs**. 
3. **Clean the data**. Handle missing values and transform the data into a more usable format.
3. **Train the model**. I will try out a few different models, including linear regression, etc. 
4. **Evaluate the model**. For each model trained, we will evaluate the model accuracy. 

Now, let's start diving into the data!

"""
## load packages
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')


os.chdir('/Users/yifeiliu/Desktop/HousePrice')
os.getcwd()
## read data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print('training dataset has %d rows and %d columns' % df_train.shape)
print('test dataset dimension has %d rows and %d columns' % df_test.shape)

Id_train = df_train['Id']
Id_test = df_test['Id']

## set aside the sales price for training set
y_train = df_train[['Id', 'SalePrice']]

## append training and testing independent variables
X = df_train.iloc[:,0:80]
X = X.append(df_test.iloc[:,0:80])

## print out independent variable names and dimension of data
print('Variables are:')
print(list(X))

missing = pd.DataFrame({'is_null': df_test.isnull().any()})
missing = missing.loc[lambda df: df.is_null == True, :].index.tolist()

## drop the variables that have missing in testing dataset
X = X.drop(missing, axis = 1)

missing = pd.DataFrame({'is_null': X.isnull().any()})
missing = missing.loc[lambda df: df.is_null == True, :].index.tolist()

for var in missing:
	print("Missing variable: " + str(var) )
	print(X[var].count())

## fill the missing with mode, running below line I could see that mode of Eletrical is SBrkr
# X.groupby("Electrical").Id.count()
X.Electrical = X.Electrical.fillna("SBrkr")
X.info()


"""
Besides the "Id" variable and dependent variable "SalePrice", we have 46 variables can be used as model inputs. We could categorize all these variables into three type, "categorical", "numeric" and "date". Then let's see some basic descriptions about these variables.
"""
## put these variables into three categories
date_l = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']

numeric_l = ['LotArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 
             'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

cat_l = ['MSSubClass', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
         'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'ExterQual', 
         'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'FullBath', 'HalfBath', 
         'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'PavedDrive', 'SaleCondition']

## create HouseAge and RemodAge
X['HouseAge'] = np.maximum(X['YrSold'] - X['YearBuilt'], np.zeros(X.shape[0]))
X['RemodAge'] = np.maximum(X['YrSold'] - X['YearRemodAdd'], np.zeros(X.shape[0]))

X = X.drop(date_l, axis = 1)

for var in cat_l:
	X[var] = X[var].astype('category')

## plot a histgram
y_train['logprice'] = np.log1p(y_train['SalePrice'])
sns.distplot(y_train['logprice'], bins=300, fit=stats.norm)
plt.show()

x = X.iloc[0:1460, ]
sns.scatterplot(x['GrLivArea'], y_train['logprice'])
plt.show()

Id_delete = [524, 1299]
X = X.loc[lambda X: X.Id != 524, :]
X = X.loc[lambda X: X.Id != 1299, :]

y_train = y_train.loc[lambda y_train: y_train.Id != 524, :]
y_train = y_train.loc[lambda y_train: y_train.Id != 1299, :]

# Ridge Regression
# We will trian the regression model using inputs from X. First, convert all categorical varaibles into dummy variables and drop the first dummy variable to avoid multicollinearity. And then split the test and train dataset. Create a pipeline that has StandardScaler, to standardize all variables, and Ridge, to perform ridge regression. Then use GridSearchCV to find the best ridge regression tuning parameter, alpha. After training model, we use the model to predict sales price on training dataset and do some diagnosis on residual.

X1 = pd.get_dummies(X, drop_first = True)
X_train = X1.loc[X1['Id'].isin(Id_train), :]
X_test = X1.loc[X1['Id'].isin(Id_test), :]

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

steps = [('scaler', StandardScaler()), ('ridge', Ridge())]

pipeline = Pipeline(steps)

params = {'ridge__alpha': np.logspace(2, 3, 100)}

cv = GridSearchCV(pipeline, params, cv=4)
cv.fit(X_train, y_train['logprice'])
y_train['pred'] = cv.predict(X_train)

"""rmse on the train dataset"""
rmse = np.sqrt(mean_squared_error(y_train['logprice'], y_train['pred']))
print("Root Mean Squared Error: {}".format(rmse))

"""plot residual"""
y_train['e'] = y_train['logprice'] - y_train['pred']
sns.distplot(y_train['e'], bins=300, fit=stats.norm)
plt.show()

sns.relplot(x='logprice', y='e', data=y_train)
plt.show()

"""gradient tree model"""

X2 = X.copy()
for var in cat_l:
	X2[var] = X2[var].cat.codes

X_train = X2.loc[X2['Id'].isin(Id_train), :]
X_train = X_train.drop(columns = 'Id')
X_test = X2.loc[X2['Id'].isin(Id_test), :]
X_test = X_test.drop(columns = 'Id')
from sklearn.ensemble import GradientBoostingRegressor

params = {'gbreg__n_estimators': [1800,2000, 2200, 2400],
		  'gbreg__learning_rate': [0.01, 0.1]}

steps = [('scaler', StandardScaler()),
		('gbreg', GradientBoostingRegressor(min_samples_split=2,min_samples_leaf=10,max_features='auto',random_state=0,subsample = 0.8))]

pipeline = Pipeline(steps)

GBcv = GridSearchCV(pipeline, params, cv = 3, n_jobs = 3, scoring = 'neg_mean_squared_error', iid = False)
GBcv.fit(X_train, y_train['logprice'])

print(GBcv.best_params_)

GBR = GradientBoostingRegressor(min_samples_split=2, min_samples_leaf=10, max_features='auto', learning_rate = 0.01, n_estimators = 1800, random_state=0)

predictors = X_train.columns
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)

GBR.fit(X_train, y_train['logprice'])
feat_imp = pd.Series(GBR.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind = 'bar', title = "Importance of Features")
plt.ylabel("Feature Importance Score")
plt.show()

y_train['pred'] = GBR.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_train['logprice'], y_train['pred']))
print("Root Mean Squared Error: {}".format(rmse))

"""plot residual"""
y_train['e'] = y_train['logprice'] - y_train['pred']
sns.distplot(y_train['e'], bins = 300, fit = stats.norm)
plt.show()

sns.relplot(x='logprice', y='e', data=y_train)
plt.show()

y_pred = GBcv.predict(X_test)
y_pred = np.exp(y_pred) - 1
prediction = pd.DataFrame({'Id': Id_test,
							'SalePrice': y_pred})

prediction.to_csv("csv_to_submit.csv", index = False)



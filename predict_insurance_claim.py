import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


def filter_file(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean the data.
    Drop uninformative rows.
    '''
    df = df.dropna().reset_index(drop=True)          # drop all rows with NaN values
    rows_count = len(df.index)
    df = df[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker',	'region', 'claim']]
    return df
  
  
def fit_and_predict_degrees_tree(df: pd.DataFrame) -> None:
    '''
    Build the machine learning Regression Tree model and reture the MSE.
    '''
    features = df.loc[:, df.columns != 'claim']
    labels = df['claim']
    features = pd.get_dummies(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print('Tree MSE test set:', round(mean_squared_error(labels_test, predictions), 2))
    
    
def fit_and_predict_degrees_linear(df: pd.DataFrame) -> None:
    '''
    Build the machine learning linear regression model and reture the MSE.
    '''
    dummies = pd.get_dummies(df[['gender', 'diabetic','smoker', 'region']])
    y = df['claim']
    X_numerical = df.drop(['claim', 'gender', 'diabetic','smoker', 'region'], axis=1).astype('float64')
    list_numerical = X_numerical.columns
    X = pd.concat([X_numerical, dummies[['gender_female', 'diabetic_No', 'smoker_No', 'region_northeast', 'region_northwest', 'region_southeast']]], axis=1)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Standardization
    scaler = StandardScaler().fit(X_train[list_numerical]) 
    X_train[list_numerical] = scaler.transform(X_train[list_numerical])
    X_test[list_numerical] = scaler.transform(X_test[list_numerical])
    
    # Lasso with arbitrarily regularization parameter α=1
    reg = Lasso(alpha=1)
    reg.fit(X_train, y_train)
    Lasso(alpha=1)
    
    # model evaluation
    pred_train = reg.predict(X_train)                      # Training data
    mse_train = mean_squared_error(y_train, pred_train)
    print('Linear MSE training set:', round(mse_train, 2))
    pred = reg.predict(X_test)                             # Test data
    mse_test =mean_squared_error(y_test, pred)
    print('Linear MSE test set:', round(mse_test, 2))
    
    # Lasso with 5 fold cross-validation
    # Fit model
    model = LassoCV(cv=5, random_state=0, max_iter=10000)
    model.fit(X_train, y_train)
    LassoCV(cv=5, max_iter=10000, random_state=0)
    print('best value of penalization:', round(model.alpha_, 2))
    # Best model
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(X_train, y_train)
    print(list(zip(lasso_best.coef_, X)))
    print('best lasso model test set mse:', round(mean_squared_error(y_test, lasso_best.predict(X_test)), 2))
    
    
def main():
    data = pd.read_csv('insurance_data.csv')
    # Call functions here
    data = filter_file(data)
    fit_and_predict_degrees_tree(data)
    fit_and_predict_degrees_linear(data)

if __name__ == '__main__':
    main()
    

    '''
    Tree MSE test set: 62462237.38
    Linear MSE training set: 43396839.03
    Linear MSE test set: 41346465.04
    best value of penalization: 40.48
    [(220.67777789171802, 'age'), (1973.9114357246187, 'bmi'), (2578.4219804815343, 'bloodpressure'), (696.9285432747407, 'children'),
    (-0.0, 'gender_female'), (15.758597983817875, 'diabetic_No'), (-20670.70474042754, 'smoker_No'), (1918.4386309856318, 'region_northeast'),
    (0.0, 'region_northwest'), (-477.0371559518968, 'region_southeast')]
    best lasso model mse: 41401657.36
    '''

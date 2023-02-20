import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.linear_model import LinearRegression


def filter_file(df: pd.DataFrame):
    '''
    Clean the data.
    Drop uninformative rows.
    '''
    df = df.dropna().reset_index(drop=True)          # drop all rows with NaN values
    rows_count = len(df.index)
    df = df[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker',	'region', 'claim']]

    
def fit_and_predict_degrees_tree(df: pd.DataFrame) -> float:
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
    return mean_squared_error(labels_test, predictions)
   

def fit_and_predict_degrees_linear(df: pd.DataFrame) -> float:
    '''
    Build the machine learning linear regression model and reture the MSE.
    '''
    X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)
    
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    
    
    
    
    
    


def main():
    data = pd.read_csv('insurance_data.csv', na_values=['---'])
    # Call functions here
    filter_file(data)
    tree_mse = fit_and_predict_degrees_tree(data)
    linear_mse = fit_and_predict_degrees_linear(data)


if __name__ == '__main__':
    main()

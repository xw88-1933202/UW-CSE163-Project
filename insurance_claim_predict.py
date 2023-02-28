import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
    df['diabetic'] = df['diabetic'].map({'No': 0, 'Yes': 1})
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


def fit_and_predict_degrees_linear(df: pd.DataFrame) -> list:
    '''
    Build the machine learning linear regression model and reture the MSE.
    '''
    features = df.loc[:, df.columns != 'claim']
    labels = df['claim']
    features = pd.get_dummies(features)
    X_numerical = df.drop(['claim', 'gender', 'diabetic','smoker', 'region'], axis=1).astype('float64')
    list_numerical = X_numerical.columns

    # split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

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

    return [X_train, y_train, X_test, y_test]


def cross_validation(df: pd.DataFrame, train_list: list) -> None:
    '''
    Use cross validation to calculate the best α and build the best model
    '''
    features = df.loc[:, df.columns != 'claim']
    features = pd.get_dummies(features)
    # Lasso with 5 fold cross-validation
    model = LassoCV(cv=5, random_state=0, max_iter=10000)
    X_train = train_list[0]
    y_train = train_list[1]
    X_test = train_list[2]
    y_test = train_list[3]
    # Fit model
    model.fit(X_train, y_train)
    LassoCV(cv=5, max_iter=10000, random_state=0)
    print('best value of penalization:', round(model.alpha_, 2))
    # Best model
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(X_train, y_train)
    print(list(zip(lasso_best.coef_, features)))
    print('best lasso model mse:', round(mean_squared_error(y_test, lasso_best.predict(X_test)), 2))


def fit_and_predict_diabetic(df: pd.DataFrame) -> float:
    '''
    Build the machine learning Classification Tree model and reture the accuracy score.
    '''
    features = df.loc[:, df.columns != ('diabetic')]
    features = df.loc[:, ~df.columns.isin(['diabetic', 'children', 'region'])]
    labels = df['diabetic']
    features = pd.get_dummies(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=2)
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    return accuracy_score(labels_test, predictions)


def hyperparameter_tuning(df: pd.DataFrame) -> float:
    '''
    Build the machine learning Classification Tree model and return the accuracy score.
    '''
    # Define parameter possibilities as lists
    p_criterion = ['gini', 'entropy']
    p_splitter = ['best', 'random']
    p_max_depth = [1, 10, 100, 1000]
    # The scores will go here
    results = []

    # Nested loops - we need to test for all combinations
    for criterion in p_criterion:
        for splitter in p_splitter:
            for max_depth in p_max_depth:
                features = df.loc[:, ~df.columns.isin(['diabetic', 'region', 'children'])]
                labels = df['diabetic']
                features = pd.get_dummies(features)
                features_train, features_test, labels_train, labels_test = \
                    train_test_split(features, labels, test_size=0.2, random_state=2)
                # Train the model
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    splitter=splitter,
                    max_depth=max_depth
                )
                model.fit(features_train, labels_train)
                preds = model.predict(features_test)
                # Append current results
                results.append({
                    'Accuracy': round(accuracy_score(labels_test, preds), 5),
                    'P_Criterion': criterion,
                    'P_Splitter': splitter,
                    'P_MaxDepth': max_depth
                })

    # Convert to Pandas DataFrame and sort descendingly by accuracy
    results = pd.DataFrame(results)
    results = results.sort_values(by='Accuracy', ascending=False)

    return results


def main():
    data = pd.read_csv('insurance_data.csv')
    # Call functions here
    data = filter_file(data)
    fit_and_predict_degrees_tree(data)
    train_list = fit_and_predict_degrees_linear(data)
    cross_validation(data, train_list)
    fit_and_predict_diabetic(data)
    hyperparameter_tuning(data)


if __name__ == '__main__':
    main()

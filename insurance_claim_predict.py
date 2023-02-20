import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def filter_file(df: pd.DataFrame):
    '''
    Clean the data.
    Drop uninformative rows.
    '''
    df = df.dropna().reset_index(drop=True)          # drop all rows with NaN values
    rows_count = len(df.index)

    
def fit_and_predict_degrees(df: pd.DataFrame) -> float:
    '''
    Build the machine learning model and reture the MSE.
    '''
    df = df[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker',	'region', 'claim']]
    features = df.loc[:, df.columns != 'claim']
    labels = df['claim']
    features = pd.get_dummies(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    return mean_squared_error(labels_test, predictions)
   






def main():
    data = pd.read_csv('insurance_data.csv', na_values=['---'])
    # Call functions here
    filter_file(data)
    fit_and_predict_degrees(data)


if __name__ == '__main__':
    main()

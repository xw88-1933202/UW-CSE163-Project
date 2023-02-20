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
    

   






def main():
    data = pd.read_csv('insurance_data.csv', na_values=['---'])
    # Call functions here
    filter_file(data)


if __name__ == '__main__':
    main()

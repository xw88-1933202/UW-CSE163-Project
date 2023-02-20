import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def compare_bachelors_1980(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Takes the data and computes the percentages of men and women
    who achieved a minimum degree of a Bachelor’s degree in 1980.
    Return a 2-by-2 DataFrame with rows corresponding to men and
    women and columns corresponding to Sex and Total. The order
    of the rows doesn’t matter.
    '''
    is_1980 = df['Year'] == 1980
    male_and_female = df['Sex'] != 'A'
    is_bachelor = df['Min degree'] == "bachelor's"
    filtered = df[is_1980 & male_and_female & is_bachelor]
    return filtered[['Sex', 'Total']]






def main():
    '''
    Loads in the dataset provided and calls all the 6
    functions above. For all of the method calls, rely on
    any default parameters we specified.
    '''
    data = pd.read_csv('nces-ed-attainment.csv', na_values=['---'])
    # Call your functions here
    compare_bachelors_1980(data)
    top_2_2000s(data)
    line_plot_bachelors(data)
    bar_chart_high_school(data)
    plot_hispanic_min_degree(data)
    fit_and_predict_degrees(data)


if __name__ == '__main__':
    main()

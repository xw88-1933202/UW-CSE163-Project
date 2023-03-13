import pandas as pd
from diabetic import filter_file
from diabetic import fit_and_predict_diabetic, hyperparameter_tuning


# Test data
def test_filter_file(data):
    """
    Test if filter_file function returns a DataFrame with the expected columns.
    """
    result = filter_file(data)
    assert all(col in result.columns for col in ['age', 'gender', 'bmi',
                                                 'bloodpressure', 'diabetic',
                                                 'smoker', 'claim'])


def test_fit_and_predict_diabetic(data):
    """
    Test if fit_and_predict_diabetic function returns a float between 0 and 1.
    """
    result = fit_and_predict_diabetic(data)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_hyperparameter_tuning(data):
    """
    Test if hyperparameter_tuning function returns a DataFrame
    with the expected columns.
    """
    result = hyperparameter_tuning(data)
    assert all(col in result.columns for col in ['Accuracy', 'P_Criterion',
                                                 'P_Splitter', 'P_MaxDepth'])


def main():
    """
    Main function that applies the functions to the insurance
    data file named "insurance_data.csv"
    and runs tests on the results.
    """
    data = pd.read_csv('insurance_data.csv')
    test_data = filter_file(data)
    test_filter_file(test_data)
    test_fit_and_predict_diabetic(test_data)
    test_hyperparameter_tuning(test_data)


if __name__ == '__main__':
    main()

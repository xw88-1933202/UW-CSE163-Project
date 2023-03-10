import pandas as pd
import diabetic

# Test data
def test_filter_file():
    # Test if function returns a DataFrame with the expected columns
    result = filter_file(data)
    assert all(col in result.columns for col in ['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'smoker', 'claim'])

def test_fit_and_predict_diabetic():
    # Test if function returns a float between 0 and 1
    result = fit_and_predict_diabetic(data)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_hyperparameter_tuning():
    # Test if function returns a DataFrame with the expected columns
    result = hyperparameter_tuning(data)
    assert all(col in result.columns for col in ['Accuracy', 'P_Criterion', 'P_Splitter', 'P_MaxDepth'])


def main():
    df = pd.read_csv('insurance_data.csv')
    test_df = filter_file(df)
    test_filter_file(test_df)
    test_fit_and_predict_diabetic(test_df)
    test_hyperparameter_tuning(test_df)
    

if __name__ == '__main__':
    main()
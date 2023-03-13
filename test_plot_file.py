import pandas as pd
from diabetic import filter_file

# Test if data is pre-processed correctly
def test_filter_file():
    # Load the dataset
    df = pd.read_csv("test_small_data.csv")

    # Filter the dataset
    filtered_df = filter_file(df)

    # Check if the filtered dataset has the expected number of rows
    assert len(filtered_df) == 2

    # Check if the filtered dataset has the expected number of columns
    assert len(filtered_df.columns) == 9

    # Check if the filtered dataset does not contain any NaN values
    assert filtered_df.isnull().sum().sum() == 0

def main():
    test_filter_file()

if __name__ == '__main__':
    main()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def filter_file(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean the data.
    Drop uninformative rows.
    '''
    # drop all rows with NaN values
    df = df.dropna().reset_index(drop=True)
    df = df[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic',
             'children', 'smoker',	'region', 'claim']]
    df['diabetic'] = df['diabetic'].map({'No': 0, 'Yes': 1})
    return df


def fit_and_predict_diabetic(df: pd.DataFrame) -> float:
    '''
    Build the machine learning Classification Tree
    model and reture the accuracy score.
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
    score = accuracy_score(labels_test, predictions)
    return score


def hyperparameter_tuning(df: pd.DataFrame) -> float:
    '''
    Build the machine learning Classification Tree model by taking
    different hyperparameters and return the accuracy score.
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
                features = df.loc[:, ~df.columns.isin(['diabetic', 'region',
                                                       'children'])]
                labels = df['diabetic']
                features = pd.get_dummies(features)
                features_train, features_test, labels_train, labels_test = \
                    train_test_split(features, labels, test_size=0.2,
                                     random_state=2)
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
    fit_and_predict_diabetic(data)
    hyperparameter_tuning(data)


if __name__ == '__main__':
    main()

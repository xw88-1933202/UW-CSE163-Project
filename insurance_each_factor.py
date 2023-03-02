import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def filter_file(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean the data.
    Drop uninformative rows.
    '''
    df = df.dropna().reset_index(drop=True)          # drop all rows with NaN values
    rows_count = len(df.index)
    df = df[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker',	'region', 'claim']]
    # apply mapping to 'diabetic' column
    df['diabetic'] = df['diabetic'].map({'Yes': 1, 'No': 0})
    # apply mapping to 'smoker' column
    df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})
    return df

def percent_diabeic(df: pd.DataFrame) -> pd.DataFrame:
    df['diabetic'] = pd.to_numeric(df['diabetic'], errors='coerce')
    # calculate proportion of diabetic individuals in each region
    prop_dia = df.groupby('region')['diabetic'].mean()
    return prop_dia

def percent_smoker(df: pd.DataFrame) -> pd.DataFrame:
    df['smoker'] = pd.to_numeric(df['smoker'], errors='coerce')
    # calculate proportion of smoker individuals in each region
    prop_smoker = df.groupby('region')['smoker'].mean()
    return prop_smoker

def dia_amount(df: pd.DataFrame) -> pd.DataFrame:
    # claim related the diabetic by region
    filter_dia = df[df['diabetic'] == 1]
    ave_dia_claim = filter_dia.groupby('region')['claim'].mean()
    ave_dia_claim
    # claim related the no diabetic by region
    filter_no_dia = df[df['diabetic'] == 0]
    ave_no_dia_claim = filter_no_dia.groupby('region')['claim'].mean()
    return (ave_dia_claim, ave_no_dia_claim)

def smoker_amount(df: pd.DataFrame) -> pd.DataFrame:
    # claim related the smoker by region
    filter_smoker = df[df['smoker'] == 1]
    ave_smoker_claim = filter_smoker.groupby('region')['claim'].mean()
    # claim related the no smoker by region
    filter_no_smoker = df[df['smoker'] == 0]
    ave_no_smoker_claim = filter_no_smoker.groupby('region')['claim'].mean()
    return (ave_smoker_claim, ave_no_smoker_claim)

def ave_amount(df: pd.DataFrame) -> pd.DataFrame:
    ave_claim = df.groupby('region')['claim'].mean()
    return ave_claim
    
def plot_percentage(df: pd.DataFrame) -> None:
    prop_dia = percent_diabeic(df)
    prop_smoker = percent_smoker(df)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    prop_dia.plot(kind='bar', ax=ax[0])
    ax[0].set_title('Percentage of Diabetics by Region')
    ax[0].set_xlabel('Region')
    ax[0].set_ylabel('Percentage')

    prop_smoker.plot(kind='bar', ax=ax[1])
    ax[1].set_title('Percentage of Smokers by Region')
    ax[1].set_xlabel('Region')
    ax[1].set_ylabel('Percentage')

    fig.tight_layout()
    plt.show()

def plot_claim(df: pd.DataFrame) -> None:  
    fig, axs = plt.subplots(ncols=3, figsize=(15,5))
    
    # Plot 1: Average claim amount for diabetic and non-diabetic individuals
    ave_dia_claim, ave_no_dia_claim = dia_amount(df)
    dia_data = pd.DataFrame({'Diabetic': ave_dia_claim, 'Non-Diabetic': ave_no_dia_claim})
    sns.barplot(data=dia_data, ax=axs[0])
    axs[0].set_title('Average Claim Amount by Diabetic Status')
    
    # Plot 2: Average claim amount for smoker and non-smoker individuals
    ave_smoker_claim, ave_no_smoker_claim = smoker_amount(df)
    smoker_data = pd.DataFrame({'Smoker': ave_smoker_claim, 'Non-Smoker': ave_no_smoker_claim})
    sns.barplot(data=smoker_data, ax=axs[1])
    axs[1].set_title('Average Claim Amount by Smoker Status')
    
    # Plot 3: Average claim amount by region
    ave_claim = ave_amount(df)
    sns.barplot(x=ave_claim.index, y=ave_claim.values, ax=axs[2])
    axs[2].set_title('Average Claim Amount by Region')
    axs[2].set_xlabel('Region')
    
    plt.tight_layout()
    plt.show()

def main():
    data = pd.read_csv('insurance_data.csv')
    # Call functions here
    percent_diabeic(data)
    percent_smoker(data)
    dia_amount(data)
    smoker_amount(data)
    ave_amount(data)
    plot_percentage(data)
    plot_claim(data)
    

if __name__ == '__main__':
    main()

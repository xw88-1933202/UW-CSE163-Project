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
    ave_claim = df.groupby('region')['claim'].agg(['mean', 'count']).reset_index()
    return ave_claim

def plot_claim(df: pd.DataFrame) -> None:  
    ave_claim = ave_amount(df)
    # Plot pie chart for average claim amount by region
    fig = px.pie(ave_claim, values='mean', names='region', title='Average Claim Amount by Region')
    fig.show()

def plot_gender_factor(df: pd.DataFrame) -> None:
    df['is_smoker'] = df['smoker'].apply(lambda x: 'Smoker' if x == 'yes' else 'Non-Smoker')
    df['is_gender'] = df['gender'].apply(lambda x: 'Male' if x == 'male' else 'Female')
    # Group by region, smoking status, and gender, and calculate the average coverage
    ave_amount = df.groupby(['region', 'is_smoker', 'is_gender']).agg({'claim': 'mean'}).reset_index()
    # start Dash
    app = dash.Dash(__name__)
    # add the bar - region
    region_options = [{'label': region, 'value': region} for region in df['region'].unique()]

    
    app.layout = html.Div(children=[
    html.H1(children='Average Claim by Region in Gender'),
        
    dcc.Dropdown(
        id='region-dropdown',
        options=region_options,
        value=df['region'].unique()[0]
    ),

    dcc.Graph(
        id='smoker-male',
    ),

    dcc.Graph(
        id='smoker-female',
    ),

    dcc.Graph(
        id='non-smoker-male',
    ),

    dcc.Graph(
        id='non-smoker-female',
    )
])

    
    # Define callback to update graphs when region dropdown is changed
    @app.callback(
        [dash.dependencies.Output('smoker-male', 'figure'),
         dash.dependencies.Output('smoker-female', 'figure'),
         dash.dependencies.Output('non-smoker-male', 'figure'),
         dash.dependencies.Output('non-smoker-female', 'figure')],
        [dash.dependencies.Input('region-dropdown', 'value')]
    )
    def update_graphs(region):
        # Filter DataFrame by selected region
        df_region = df[df['region'] == region]

        # Group by region, smoking status, and gender, and calculate the average coverage
        avg_amount = df_region.groupby(['region', 'is_smoker', 'is_gender']).agg({'claim': 'mean'}).reset_index()

        # Create figures for each graph
        fig_smoker_male = px.bar(avg_coverage[(avg_coverage['is_smoker'] == 'Smoker') & (avg_coverage['is_gender'] == 'Male')], 
                          x='region', y='claim', color='is_smoker', barmode='group')
        fig_smoker_female = px.bar(avg_coverage[(avg_coverage['is_smoker'] == 'Smoker') & (avg_coverage['is_gender'] == 'Female')], 
                          x='region', y='claim', color='is_smoker', barmode='group')
        fig_non_smoker_male = px.bar(avg_coverage[(avg_coverage['is_smoker'] == 'Non-Smoker') & (avg_coverage['is_gender'] == 'Male')], 
                          x='region', y='claim', color='is_smoker', barmode='group')
        fig_non_smoker_female = px.bar(avg_coverage[(avg_coverage['is_smoker'] == 'Non-Smoker') & (avg_coverage['is_gender'] == 'Female')], 
                          x='region', y='claim', color='is_smoker', barmode='group')

        # Return figures for each graph
        return fig_smoker_male, fig_smoker_female, fig_non_smoker_male, fig_non_smoker_female
    app.run_server(debug=True)

def main():
    df = pd.read_csv('insurance_data.csv')
    # Call functions here
    percent_diabeic(df)
    percent_smoker(df)
    dia_amount(df)
    smoker_amount(df)
    ave_amount(df)
    plot_claim(df)
    plot_gender_factor(df)
    

if __name__ == '__main__':
    main()

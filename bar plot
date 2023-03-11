import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from diabetic import filter_file

# Load and filter the data
df = pd.read_csv("insurance_data.csv")
df = filter_file(df)

# Group the data by 'region' and 'gender', and calculate the mean claim for each group
grouped_data = df.groupby(['region', 'gender']).mean()['claim'].reset_index()

# Create the app and layout
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='insurance-plot',
        figure={
            'data': [
                {'x': grouped_data[grouped_data['gender'] == gender]['region'],
                 'y': grouped_data[grouped_data['gender'] == gender]['claim'],
                 'type': 'bar', 'name': gender} for gender in grouped_data['gender'].unique()
            ],
            'layout': {
                'title': 'Mean Claim by Region and Gender',
                'xaxis': {'title': 'Region'},
                'yaxis': {'title': 'Mean Claim'},
                'barmode': 'group'
            }
        }
    ),
    dcc.Dropdown(
        id='gender-dropdown',
        options=[{'label': gender, 'value': gender} for gender in grouped_data['gender'].unique()],
        value=grouped_data['gender'].unique()[0]
    )
])

@app.callback(
    Output('insurance-plot', 'figure'),
    [Input('gender-dropdown', 'value')]
)
def update_plot(gender):
    filtered_data = grouped_data[grouped_data['gender'] == gender]
    figure={
        'data': [
            {'x': filtered_data[filtered_data['region'] == region]['region'],
             'y': filtered_data[filtered_data['region'] == region]['claim'],
             'type': 'bar', 'name': region} for region in filtered_data['region'].unique()
        ],
        'layout': {
            'title': 'Mean Claim by Region and Gender',
            'xaxis': {'title': 'Region'},
            'yaxis': {'title': 'Mean Claim'},
            'barmode': 'group'
        }
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

# Load the dataset
df = pd.read_csv('insurance_data.csv')

# Define the app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='scatter-plot',
              figure=px.scatter(df, x='bmi', y='claim', color='gender',
                                trendline='ols', custom_data=['gender']))
])


@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('gender-dropdown', 'value')])
def update_figure(selected_gender):
    filtered_df = df[df['gender'] == selected_gender]
    fig = px.scatter(filtered_df, x='bmi', y='claim', color='gender',
                     trendline='ols', custom_data=['gender'])
    fig.update_traces(marker=dict(
        color=['blue' if gender == 'male' else 'red'
               for gender in filtered_df['gender']]
    ))
    return fig


# Add the dropdown menu to the app layout
app.layout = html.Div([
    dcc.Dropdown(id='gender-dropdown', options=[{'label': 'Male',
                                                 'value': 'male'},
                                                {'label': 'Female',
                                                'value': 'female'}],
                 value='male'),
    dcc.Graph(id='scatter-plot', figure=px.scatter(df, x='bmi', y='claim',
                                                   color='gender',
                                                   trendline='ols'))
])
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

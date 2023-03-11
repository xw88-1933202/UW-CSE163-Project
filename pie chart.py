import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

df = pd.read_csv('insurance_data.csv')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Insurance Claims by Region'),
    dcc.Dropdown(
        id='region-dropdown',
        options=[{'label': r, 'value': r} for r in df['region'].dropna().unique()],
        value=df['region'].dropna().unique()[0]
    ),
    dcc.Graph(id='pie-chart')
])

@app.callback(
    dash.dependencies.Output('pie-chart', 'figure'),
    [dash.dependencies.Input('region-dropdown', 'value')]
)
def update_pie_chart(region):
    data = df[df['region'] == region]
    fig = px.pie(data, values='claim', names='gender')
    fig.update_traces(hoverinfo='label+percent', textinfo='value+label')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
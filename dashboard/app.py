import dash
from dash import html

app = dash.Dash(__name__)
server = app.server  # Needed for deployment

app.layout = html.Div([
    html.H1("Democracy Clustering Dashboard"),
    html.P("Welcome to your first Dash app!"),
])

if __name__ == "__main__":
    app.run(debug=True)



import dash
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div("Hello, world!")

if __name__ == '__main__':
    app.run_server(debug=True)
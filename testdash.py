from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd



# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

def test():
    print("I like mice")
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    return fig

def main():
    print("I like main")
    fig =test()
    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ])

app = Dash(__name__)

if __name__ == '__main__':
    main()
    app.run_server(debug=True, use_reloader=False)
""""
Without Prophet - Just TFIDF based
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table
import logoutlier_util as util

app = Dash(__name__)

LIMIT_FILE_READ = 10000000
LIMIT_FILE_PROCESS = 50000





def vectorize_df_list(df):

    # token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' - this is default with numbers
    # we are skpiping text with numbers
    print("--Going tovectorize_df_list ", df.shape[0])
    tfidf_vectorizer = TfidfVectorizer(
        token_pattern=u'(?ui)\\b[a-z][a-z-_/0-9]{3,}\\b', stop_words='english')
    tfidf_vector = tfidf_vectorizer.fit_transform(df['y_org'])
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(
    ), index=df['ds_org'], columns=tfidf_vectorizer.get_feature_names())
    tfidf_df['y'] = tfidf_df.apply(np.max, axis=1)
    df = df.assign(y=tfidf_df["y"].reset_index(drop=True))
    print("Input data size", df.shape[0])
    pd.set_option('display.max_columns', None)
    print("-----------vectorize_df head--------")
    print(df.head())
    df.to_csv("df_rank.csv")
    tfidf_df.to_csv("tfidf_rank.csv")
    return df


if __name__ == '__main__':
    """
    Given a syslog file; or any logfile with the corresponding change in processing; this program parses
    the file and reduces it - removing duplicates
    """
    print(""""
    Usage - <path to Logfile>  <outputfile path>
    """)

    if len(sys.argv) != 3:
        print("Error - Please give the argumetns")
        sys.exit()

    path_to_log = sys.argv[1]
    path_to_outfile = sys.argv[2]

    start_time = time.time()
    start_time_intial = start_time
    df = read_syslog_in_parts(sys.argv[1])
    print("---read_syslog_in_parts %s seconds ---" %
          (time.time() - start_time))
    print("Input data size = ", df.shape[0])
    start_time = time.time()

    # Limit processing of files
    if df.shape[0] > LIMIT_FILE_PROCESS:
        print("LIMITING File processing to %d records" % (LIMIT_FILE_PROCESS))
        df = df.tail(LIMIT_FILE_PROCESS)
        df = df.reset_index()

    df_new = vectorize_df_list(df)
    print("---vectorize_df_list %s seconds ---" %
          (time.time() - start_time))
    if df_new is None:
        print("Error in calculating similrity, Exiting")
        sys.exit(-1)
    start_time = time.time()

    print("---Full Runtime %s seconds ---" % (time.time() - start_time_intial))

    # ---------------------------------------
    #  Plotly & Dash based display
    # ----------------------------------------

    fig = make_subplots(rows=1, cols=1)
    fcst_t = df_new['ds'].dt.to_pydatetime()

    # Plot all the data / historical logs first
    fig.add_trace(
        go.Scatter(x=df_new['ds'].dt.to_pydatetime(), y=df_new['y'], mode='markers',
                   name="history(all)",
                   text=df['y_org'],
                   line=dict(color='rgb(108, 122, 137)', width=2)),
        row=1, col=1
    )

    # Plot the thresholded logs that we calculated
    fig.add_trace(
        go.Scatter(x=df_new['ds'].dt.to_pydatetime(), y=df_new['y'], mode='markers',
                   name="thresholded",
                   text=df_new['y_org'],
                   line=dict(color='red', width=2)),
        row=1, col=1
    )

    app.layout = html.Div(children=[
        html.H1(children='Log Anomaly Analysis'),

        html.Div(children='''
            Used TFIDF and Prophet TimeSeries to dosplay the Thresholded Logs
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),

        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                # {"name": i, "id": i} for i in thresholded_df.loc[:,['ds_org','y_org']]
                {"name": "Log Time", "id": "ds_org"},
                {"name": "Log Message", "id": "y_org"}
            ],
            style_filter={'textAlign': 'left'},
            style_cell={'textAlign': 'left'},
            data=df_new.to_dict('records'),
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            # row_selectable="multi",
            css=[{
                'selector': '.dash-spreadsheet td div',
                'rule': '''
                    line-height: 15px;
                    max-height: 30px; min-height: 30px; height: 30px;
                    display: block;
                    overflow-y: hidden;
                   ''',
                'selector': '.dash-filter input',
                'rule': '''
                    text-align: left !important;
                    padding-left: 5px !important;
                '''
            }],
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=100,
            style_table={'overflowX': 'auto'}
        ),
        html.Div(id='datatable-interactivity-container'),

    ])
    app.run_server(debug=True, use_reloader=False)

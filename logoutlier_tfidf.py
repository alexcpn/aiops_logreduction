"""
This module pares a linux syslog file (can be extended to any log file) and 
    1. Split each log line to a set of terms (vectorize)
    2. Uses Term Frequency Inverse Document Frequency rank to rank each term in the line
    3. Uses a algorithm to pick a representative rank for each line
    4. Uses the ranks to cluster similar logs 
    5. Removes the duplicate logs with same ranks
    6. Uses Time-series forecasting library to 'fit' the de-duplicated logs against time plot
    7. Thresholds all the logs which do not fall inside the time forecast fit
    8. Uses dash and plotly to display visually the de-deuplicated logs and thresholded logs in Browser

"""
from pickle import FALSE
import pandas as pd
from prophet import Prophet
import numpy as np
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table
import logoutlier_util as util

__author__ = "alex.punnen@nokia.com"


# Dash app name
app = Dash(__name__)

# Limits for file read and processing / ex laptop use
LIMIT_FILE_READ = 10000000
LIMIT_FILE_PROCESS = 100000
FIT_TIME_SERIES = False


def vectorize_df(df, outpath):

    # token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' - this is default with numbers
    # we are skpiping text with numbers
    print("--Going tovectorize_df_list ", df.shape[0])
    
    # Where vectorization happens
    tfidf_vectorizer = TfidfVectorizer(
        # token_pattern=u'(?ui)\\b[a-z][a-z-_/0-9]{3,8}\\b', stop_words='english')
        # trying without numbers
        token_pattern=u'(?ui)\\b[a-z][a-z-_/]{3,8}\\b', stop_words='english')

    tfidf_vector = tfidf_vectorizer.fit_transform(df['y_org'])

    # Het the TFIDF terms as column headers and rank of each term as dataframe
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(
    ), index=df['ds_org'], columns=tfidf_vectorizer.get_feature_names())
    
    # Rank each row based on a Rank - here we take the max of the coulumns as rank
    tfidf_df['y'] = tfidf_df.apply(np.max, axis=1)

    # Assing this Rank column to original data fram

    df = df.assign(y=tfidf_df["y"].reset_index(drop=True))
    print("Input data size", df.shape[0])
    
    pd.set_option('display.max_columns', None)
    print("-----------vectorize_df head--------")
    print(df.head())
    df.to_csv(outpath + "/df_rank.csv")
    tfidf_df.to_csv(outpath+"/tfidf_rank.csv")
    return df


def remove_duplicates(df):
    """
    Remove the duplicates of the data-frame
    """
    df.y = df.y.round(5)

    # create a new column to keep track of count of hits
    frequency = df.y.value_counts().rename_axis('y').reset_index(name='counts')
    print(frequency.head())
    df = df.merge(frequency, on='y', how='outer')
    df = df.drop_duplicates(subset=['y'], keep='first')
    return df


def fit_timeseries(df):
    """
       Fit the log data in timeseries graph
    """
    df = df.reset_index()
    print("fit_timeseries---Data Length=", df.shape[0])
    print("df.tail()")
    print(df.tail())

    m = Prophet(changepoint_prior_scale=0.5, changepoint_range=1,
                uncertainty_samples=100, interval_width=.95)
    # uncertainty_samples default is 1000
    # changepoint_prior_scale default is 0.05
    print("Going to fit the Data")
    m.fit(df)

    # we are not concerned about predicting here, rather just fitting the data
    # https://stackoverflow.com/a/35339226/429476
    # future = m.make_future_dataframe(periods=2, freq='us', include_history=True)
    future = m.make_future_dataframe(periods=2, freq='T', include_history=True)
    # print(future.tail())
    forecast = m.predict(future)

    print("forecast length", forecast.shape[0])
    print("forecast.tail()")
    print(forecast.tail())

    # Lets identify the points that are over the threshold
    # find the dataframes having same indices
    forecast_truncated_index = forecast.index.intersection(df.index)
    forecast_truncated = forecast.loc[forecast_truncated_index]

    # truncate the history also to fit prediction
    m.history = m.history.loc[forecast_truncated_index]

    print("len of forecast_truncated and df =",
          forecast_truncated.shape[0], df.shape[0])

    print("forecast_truncated.tail()")
    print(forecast_truncated.tail())

    print("m.history.tail()")
    print(m.history.tail())

    # Identify the thresholds
    buffer_max = forecast_truncated['yhat_upper']
    buffer_min = forecast_truncated['yhat_lower']

    indices_max = m.history[m.history['y'] > buffer_max].index
    indices_min = m.history[m.history['y'] < buffer_min].index
    indices = indices_min.union(indices_max)
    # Get those points that have crossed the threshold
    # ------> This has the thresholded values and more important timestamp
    thresholded_df = m.history.iloc[indices]
    return thresholded_df, m, forecast


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
    df = util.read_syslog_in_parts(sys.argv[1])
    #df = util.read_csv(sys.argv[1])
    print("---read_syslog_in_parts %s seconds ---" %
          (time.time() - start_time))
    print("Input data size = ", df.shape[0])
    start_time = time.time()

    # Limit processing of files
    if df.shape[0] > LIMIT_FILE_PROCESS:
        print("LIMITING File processing to %d records" % (LIMIT_FILE_PROCESS))
        df = df.tail(LIMIT_FILE_PROCESS)
        df = df.reset_index()

    df_new = vectorize_df(df,path_to_outfile)
    vectorize_df_list_time = time.time() - start_time
    print("---vectorize_df_list %s seconds ---" %
          (vectorize_df_list_time))
    if df_new is None:
        print("Error in calculating similrity, Exiting")
        sys.exit(-1)

    """ # todo test - processing in parts
    PROCESSING_SIZE = 10000
    chunk_size = df_new.shape[0] // PROCESSING_SIZE
    if chunk_size == 0:
        chunk_size = 1
    thresholded_df, m, forecast = proceess_in_chunks(df_new, chunk_size)
    """

    df_new = remove_duplicates(df_new)
    fit_timeseries_time = 0

    if FIT_TIME_SERIES:
        start_time = time.time()
        thresholded_df, m, forecast = fit_timeseries(df_new)
        fit_timeseries_time = time.time() - start_time
        print("---fit_timeseries %s seconds ---" % (fit_timeseries_time))

    full_run_time = time.time() - start_time_intial
    print("---Full Runtime %s seconds ---" % (full_run_time))

    if FIT_TIME_SERIES:
        # Write the thresholded data out to file
        thresholded_df.to_csv(path_to_outfile + "/thresholded_logs.csv", columns=["ds_org", "counts", "y", "y_org"])

    # Write the All log  data out to file
    df_new.to_csv(path_to_outfile + "/unique_logs.csv", columns=["ds_org", "counts", "y", "y_org"])

    # ---------------------------------------
    #  Plotly & Dash based display
    # ----------------------------------------

    fig = make_subplots(rows=1, cols=1)

    if FIT_TIME_SERIES:

        fcst_t = forecast['ds'].dt.to_pydatetime()

        # Plot all the data / historical logs first
        fig.add_trace(
            go.Scatter(x=m.history['ds'].dt.to_pydatetime(), y=m.history['y'], mode='markers',
                       name="history(all)",
                       text=df['y_org'],
                       line=dict(color='rgb(108, 122, 137)', width=2)),
            row=1, col=1
        )

        # Plot the forecast
        fig.add_trace(
            go.Scatter(x=fcst_t, y=forecast['yhat'], fillcolor='grey',
                       name="forecast"),
            row=1, col=1
        )
        # Plot the upper bound of the fit that Prophet found
        fig.add_trace(
            go.Scatter(x=fcst_t, y=forecast['yhat_lower'], mode='lines',
                       name="yhat_lower",
                       line=dict(color='rgb(108, 122, 137)', width=2)),
            row=1, col=1
        )
        # Plt the lower bound of the fit that Prophet found
        fig.add_trace(
            go.Scatter(x=fcst_t, y=forecast['yhat_upper'], fillcolor='rgba(189, 195, 199, .5)', fill="tonextx", mode='lines',
                       name="yhat_upper",
                       line=dict(color='rgb(108, 122, 137)', width=2)),
            row=1, col=1
        )
        # Plot the thresholded logs that we calculated
        fig.add_trace(
            go.Scatter(x=thresholded_df['ds'].dt.to_pydatetime(), y=thresholded_df['y'], mode='markers',
                       name="thresholded",
                       text=thresholded_df['y_org'],
                       line=dict(color='red', width=2)),
            row=1, col=1
        )
    else:  # No time series fitting
        # Plot all the data / historical logs first
        fig.add_trace(
            go.Scatter(x=df_new['ds'].dt.to_pydatetime(), y=df_new['y'], mode='markers',
                       name="history(all)",
                       marker_size=4+np.log2(df_new["counts"]),
                       text=df_new['y_org'],
                       line=dict(color='rgb(108, 122, 137)', width=2),
                       marker=dict(color='rgb(255, 100, 102)')),
                        
            row=1, col=1
        )

    app.layout = html.Div(children=[
        html.H1(children='Log Anomaly Analysis'),

        html.Div(children='''
            Using TFIDF and Prophet TimeSeries to reduce and thresholded Logs'''),

        html.Br(),

        html.Div([
            html.Div("Log file Name = {}".format(path_to_log)),
            html.Br(),
            html.Div("Max Line Processing Limit Set is {}".format(
                LIMIT_FILE_PROCESS)),
            html.Br(),
            html.Div("Number of logs processed = {} ".format(
                df.shape[0]), style={'fontSize': 16}),
            html.Br(),
            html.Div("vectorize_df_list_time = {} ".format(
                vectorize_df_list_time)),
            html.Br(),
            html.Div("fit_timeseries_time = {} ".format(fit_timeseries_time)),
            html.Br(),
            html.Div("full_run_time = {} ".format(full_run_time))


        ], style={'color': 'blue', 'fontSize': 14}),

        html.Br(),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),
        html.Br(),
        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                # {"name": i, "id": i} for i in thresholded_df.loc[:,['ds_org','y_org']]
                {"name": "Log Time", "id": "ds_org"},
                {"name": "Rank", "id": "y"},
                {"name": "Repetions", "id": "counts"},
                {"name": "Log Message", "id": "y_org"}

            ],
            style_filter={'textAlign': 'left'},
            style_cell={'textAlign': 'left'},
            # data=thresholded_df.to_dict('records'),
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
        html.Div(id='datatable-interactivity-container')
    ])
    app.run_server(debug=True, use_reloader=False)

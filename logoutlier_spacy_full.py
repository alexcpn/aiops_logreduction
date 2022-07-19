import pandas as pd
from prophet import Prophet
import numpy as np
from matplotlib import pyplot as plt
import sys
import spacy
from multiprocessing import Pool
import time
import jellyfish


def read_syslog(path_to_file):
    """"
     Parse syslog/journalctl into a data frame
    """
    print("Going to read Syslog", path_to_file)
    colspecs = [(0, 16), (16, -1)]
    dflist = []
    chunk_size = 50000
    for df in pd.read_fwf(path_to_file, colspecs=colspecs, skiprows=0,
                          chunk_size=chunk_size, iterator=True):
        df.columns = ["ds_org", "y_org"]
        df["y"] = 0
        # format=Mar 31 09:32:06
        df["ds"] = "2022 " + df["ds_org"]
        df["ds"] = pd.to_datetime(df["ds"], format="%Y %b %d %H:%M:%S")
        # add a microsecond to distinguish logs
        #df["ds"] = df["ds"] + pd.to_timedelta(np.random.randint(1, 10000), unit='ms')
        #df["ds"] = df["ds"] +pd.to_timedelta(np.random.randint(1,1000000,len(df)),unit='us')
        df = df.fillna(method="ffill")
        dflist.append(df)
    df = pd.DataFrame()
    df = pd.concat(dflist, ignore_index=True)
    print("Input data size", df.shape[0])
    print(df.head())
    return df


def read_parsedlog(path_to_file):
    """"
     Parse syslog/journalctl into a data frame
    """
    print("Going to read csv", path_to_file)
    colspecs = [(0, 27), (27, -1)]
    df = pd.read_fwf(path_to_file, colspecs=colspecs)
    df.columns = ["ds", "y_org"]
    df["y"] = 0
    # format=Mar 31 09:32:06
    print(df.head())
    return df


def calculate_log_similarity(df):

    print("calculate_log_similarity Input Size=",df.shape[0])
    # Let's use more cores
    pool = Pool(processes=10)
    results = []  # Output of multi Processing results
    # Let's check the similarity of one line with the next
    str1 = None
    str2 = None
    for index, row in df.iterrows():
        if str1 == None:
            str1 = row['y_org']
        else:
            str2 = row['y_org']
            # results.append(pool.apply_async(do_nlp_similarity,(str1,str2,index)))
            results.append(pool.apply_async(
                do_jaro_similarity, (str1, str2, index)))
            str1 = str2
    print("calculate_log_similarity Result Size=",len(results))
    pool.close()
    pool.join()

    for result in results:
        similarity, index = result.get()
        df.loc[index, 'y'] = similarity
    print("----------After assinging----------")
    print("---do_similarty %s seconds ---" % (time.time() - start_time))
    df.reset_index(drop=True, inplace=True)
    print(df.head())
    nans = df.isnull().sum().sum()
    if nans > 1:
        print("calculate_log_similarity Total nans", nans, "Nan Cols", df[df['ds'].isna()])
        return None
    
    return df


def do_nlp_similarity(str1, str2, counter):
    doc1 = nlp(str1)
    doc2 = nlp(str2)
    similarity = doc1.similarity(doc2)
    # print(similarity)
    return similarity, counter


def do_jaro_similarity(str1, str2, counter):
    return jellyfish.jaro_distance(str1, str2), counter


def fit_timeseries(df):
    """
       Fit the log data in timeseries graph
    """
    df = df.reset_index()
    m = Prophet(changepoint_prior_scale=0.05, changepoint_range=1
                )  # ,interval_width=.95)
    # uncertainty_samples default is 1000

    print("fit_timeseries---Data Length=", df.shape[0])
    print("df.tail()")
    print(df.tail())
    m.fit(df)

    # we are not concerned about predicting here, rather just fitting the data
    # https://stackoverflow.com/a/35339226/429476
    # future = m.make_future_dataframe(periods=2, freq='us', include_history=True)
    future = m.make_future_dataframe(periods=2, freq='S', include_history=True)
    # print(future.tail())
    forecast = m.predict(future)

    print("forecast length",forecast.shape[0])
    print("forecast.tail()")
    print(forecast.tail())

    # Lets identify the points that are over the threshold
    # find the dataframes having same indices
    forecast_truncated_index = forecast.index.intersection(df.index)
    forecast_truncated = forecast.loc[forecast_truncated_index]

    m.history = m.history.loc[forecast_truncated_index] #trunceat the history also to fit prediction

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


def repeat_reduce(df, no_of_times, chunk_size):
    """
      Repeatedly reduce the logs

    """
    for i in range(no_of_times):
        all_res = []
        for chunk in np.array_split(df, chunk_size):
            chunk_size = chunk.shape[0]
            print("Level %d Chunk size = %d" % (i, chunk_size))
            start_time = time.time()
            thresholded_df, m, forecast = fit_timeseries(chunk)
            print("---fit_timeseries %s seconds ---" % (time.time() - start_time))
            all_res.append(thresholded_df)
        df = pd.concat(all_res, ignore_index=True)
        # write each segment out
        with open(path_to_outfile + str(i), 'w') as f:
            for _, row in df.iterrows():
                f.write("%s %s\n" % (row["ds_org"], row["y_org"]))

    return df, m, forecast


if __name__ == '__main__':
    """
    Given a syslog file; or any logfile with the corresponding change in processing; this program parses
    the file and reduces it - removing duplicates
    """
    print(""""
    Usage - <path to Logfile>  <outputfile path> [ Optional -Saved model as the argument]
    """)

    # Load the pickled data frame
    if len(sys.argv) != 3:
        print("Error - Please give the argumetns")
        sys.exit()

    path_to_log = sys.argv[1]
    path_to_outfile = sys.argv[2]

    # nlp = spacy.load("en_core_web_md") #large model - similarity is worse with this
    nlp = spacy.load("en_core_web_sm")  # small model

    df = None
    df = read_syslog(path_to_log)
    print("Log file parsed - Going to check similarity")
    start_time = time.time()

    print("Log file parsed - Going to calculate sentence embedding")
    # print(df.head)
    print("Input data size = ", df.shape[0])
    # we need to limit the processing to say last 50,000 entried
    df = df.tail(50000)
    print("Trimmed Input data size = ", df.shape[0])

    start_time = time.time()
    start_time_intial = start_time
    df_new = calculate_log_similarity(df)
    print("---calculate_log_similarity %s seconds ---" %
          (time.time() - start_time))
    if df_new is None:
        print("Error in calculating similrity, Exiting")
        sys.exit(-1)
    PROCESSING_SIZE = 10000
    chunk_size = df.shape[0] // PROCESSING_SIZE
    if chunk_size == 0:
        chunk_size = 1
    # This is the main methods that used the NLP comparison to plot against a time series
    # and give out the thresholded logs.

    start_time = time.time()
    thresholded_df, m, forecast = repeat_reduce(df, 2, chunk_size)
    print("---repeat_reduce %s seconds ---" % (time.time() - start_time))

    print("---Full Runtime %s seconds ---" % (time.time() - start_time_intial))

    with open(path_to_outfile, 'a') as f:
        for _, row in thresholded_df.iterrows():
            f.write("%s %s\n" % (row["ds_org"], row["y_org"]))
    fig = m.plot(forecast)
    figsize = (10, 6)
    fig = plt.figure(facecolor='w', figsize=figsize)
    ax = fig.add_subplot(111)
    fig = m.plot(forecast, ax=ax)

    # plot the threhsolded points as red
    ax.plot(thresholded_df['ds'].dt.to_pydatetime(), thresholded_df['y'], 'r.',
            label='Thresholded data points')
    fig.savefig('./out/log_thresholded.png')

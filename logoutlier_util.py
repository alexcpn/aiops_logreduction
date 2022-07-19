
import pandas as pd
import numpy as np

# Limits for file read and processing / ex laptop use
LIMIT_FILE_READ = 10000000

def read_syslog_in_parts(path_to_file):  # todo this is not reading in parts
    """"
     Parse syslog/journalctl into a data frame 
    """
    print("--Going to read Syslog", path_to_file, pd.__version__)
    colspecs = [(0, 16), (16, -1)]
    dflist = []
    chunk_size = 10000
    with pd.read_fwf(path_to_file, colspecs=colspecs, skiprows=0, chunksize=chunk_size) as reader:
        for df in reader:
            df.columns = ["ds_org", "y_org"]
            # format=Mar 31 09:32:06
            df["ds"] = "2022 " + df["ds_org"]
            df["ds"] = pd.to_datetime(df["ds"], format="%Y %b %d %H:%M:%S")
            # add a microsecond to distinguish logs
            #df["ds"] = df["ds"] + \
            #    pd.to_timedelta(np.random.randint(
            #        1, 1000000, len(df)), unit='us')
            df = df.fillna(method="ffill")
            dflist.append(df)
            if len(dflist)*chunk_size >= LIMIT_FILE_READ:
                print("Limiting input file read to %d entries" %
                      (LIMIT_FILE_READ))
                break
        print("Input Data list size", len(dflist))
        df = pd.DataFrame()
        df = pd.concat(dflist, ignore_index=True)
        print("Input data size", df.shape[0])
        print(df.head())
    return df



def read_csv(path_to_file):
    print("Going to read CSV", path_to_file)
    dflist = []
    chunk_size = 500
    for df in pd.read_csv(path_to_file, skiprows=0,
                          chunksize=chunk_size, iterator=True):
        df["ds_org"] = df["@timestamp"]
        df["y_org"]= df["log_field.msg"]
        df["y"] = 0
        df["ds"] = ""
        df["ds"] = pd.to_datetime(df["@timestamp"], format="%B %d, %Y @ %H:%M:%S.%f")
        df = df.fillna(method="ffill")
        dflist.append(df)
    df = pd.DataFrame()
    df = pd.concat(dflist, ignore_index=True)
    print("Input data size", df.shape[0])
    print(df.head())
    return df

def proceess_in_chunks(df, chunk_size):
    """
        Repeatedly reduce the logs
    """
    thresholded_df_list = []
    m_list = []
    forecast_list = []
    for chunk in np.array_split(df, chunk_size):
        chunk_size = chunk.shape[0]
        print("Chunk size = %d" % (chunk_size))
        start_time = time.time()
        thresholded_df, m, forecast = fit_timeseries(chunk)
        print("---fit_timeseries %s seconds ---" % (time.time() - start_time))
        thresholded_df_list.append(thresholded_df)
        m_list.append(m)
        forecast_list.append(forecast)
    df = pd.concat(thresholded_df_list, ignore_index=True)
    m = pd.concat(m_list, ignore_index=True)
    forecast_list = pd.concat(forecast_list, ignore_index=True)
    return df, m, forecast_list


    # ---------------------------------------
    #  MatPlotlib displacy
    # ----------------------------------------

    # Commenting out the matplotlib part for now- it cannot display live from Docker
    """
    fig = m.plot(forecast)
    figsize = (10, 6)
    fig = plt.figure(facecolor='w', figsize=figsize)
    ax = fig.add_subplot(111)
    fig = m.plot(forecast, ax=ax)
    # plot the threhsolded points as red
    ax.plot(thresholded_df['ds'].dt.to_pydatetime(), thresholded_df['y'], 'r.',
            label='Thresholded data points')
    fig.savefig('./out/log_thresholded.png')
    plt.show()
    """

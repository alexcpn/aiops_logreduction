from ast import Index
import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import numpy as np
from matplotlib import pyplot as plt
import pickle5 as pickle
import json
from prophet.serialize import model_to_json, model_from_json
import sys


def fit_timeseries(df) :

    m = Prophet(changepoint_prior_scale=0.05,changepoint_range=1,uncertainty_samples=1000) #,interval_width=.95)
    #uncertainty_samples default is 1000
    m.fit(df)

    # we are not concerned about predicting here, rather just fitting the data
    print("---make_future_dataframe-- ")
    future = m.make_future_dataframe(periods =2,freq='1s',include_history=True) 
    #print(future.tail())

    print("--model predict-- ")
    forecast = m.predict(future)
    
    #print(forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail())

    #Lets identify the points that are over the threshold
    # find the dataframes having same indices
    forecast_truncated_index =forecast.index.intersection(df.index)
    forecast_truncated = forecast.loc[forecast_truncated_index]
    print("len of forecast_truncated and df =",forecast_truncated.shape[0],df.shape[0])
  

    # Identify the thresholds 
    buffer_max =  forecast_truncated['yhat_upper']
    buffer_min =  forecast_truncated['yhat_lower']

    indices_max =m.history[m.history['y'] > buffer_max.reset_index(drop=True)].index
    indices_min =m.history[m.history['y'] < buffer_min.reset_index(drop=True)].index
    indices =indices_min.union(indices_max)
      # Get those points that have crossed the threshold
    thresholded_df  = m.history.iloc[indices] # ------> This has the thresholded values and more important timestamp
    return thresholded_df,m ,forecast

  
def repeat_reduce(df,no_of_times,chunk_size):
  """
    Repeatedly reduce the logs

  """
  for i in range(no_of_times):
    all_res = []
    for chunk in np.array_split(df, chunk_size):
      print("Level %d Chunk size = %d" % (i,chunk.shape[0]))  
      thresholded_df,m,forecast =fit_timeseries(chunk.reset_index(drop=True))
      all_res.append(thresholded_df)
    df = pd.concat(all_res)
    # write each segment out
    with open(path_to_outfile+ str(i), 'w') as f:
      for _, row in df.iterrows():
        f.write("%s %s\n" % (row["ds_org"] ,row["y_org"]))
    
  return  df,m,forecast
    
if __name__ == '__main__':
  """
  Use the loganalysis.py generated dataframe containing log data and log comparison data
  to find outlier in logs; 
  """
  #df = pd.read_csv("./resources/History of Save failures-data-2022-04-02 11_43_03.csv")
  print(""""
    Usage - <path to picked Dataframe>  <outputfile path> [ Opetional -Saved model as the argument]
    outlier.py <path to pickled Dataframe> [serialized_model.json]
    Else just run outlier.py
  """)

  # Load the pickled data frame
  if len(sys.argv) != 3:  
     print("Error - Please give the argumetns")
     sys.exit()

  path_to_df =sys.argv[1]
  path_to_outfile =sys.argv[2]

  with open(path_to_df, "rb") as fh:
    df = pickle.load(fh)
  print("Input data")
  df = df.iloc[1:]
  print(df.head)
  df.columns=["ds_org","y_org","y","ds"]
  print(df.head)
  print ("Input data size = ",df.shape[0])

  df = df[:50000] # for test

  # reduce data-frame 
  chunk_size = df.shape[0]/10000
  thresholded_df,m,forecast =  repeat_reduce(df,2,chunk_size)

  with open(path_to_outfile, 'a') as f:
    for _, row in thresholded_df.iterrows():
      f.write("%s %s\n" % (row["ds_org"] ,row["y_org"]))
  fig = m.plot(forecast)
  figsize=(10, 6)
  fig = plt.figure(facecolor='w', figsize=figsize)
  ax = fig.add_subplot(111)
  fig = m.plot(forecast,ax=ax)

  # plot the threhsolded points as red
  ax.plot(thresholded_df['ds'].dt.to_pydatetime(), thresholded_df['y'], 'r.',
          label='Thresholded data points')
  fig.savefig('./out/log_thresholded.png')

import spacy
import pandas as pd
import numpy as np
from multiprocessing import Pool
import sys




def read_syslog(path_to_file):
    """"
     Parse syslog/journalctl into a data frame
    """
    print("Going to read Syslog",path_to_file)
    colspecs = [(0,16),(16,-1)]
    df =pd.read_fwf(path_to_file,colspecs=colspecs,skiprows=1)
    df.columns=["ds_org","y_org"]
    df["y"] = 0
    #format=Mar 31 09:32:06 
    df["ds"] = "2022 " + df["ds_org"]
    df["ds"] =  pd.to_datetime(df["ds"],format="%Y %b %d %H:%M:%S")
    # add a microsecond to distinguish logs
    df["ds"] = df["ds"] +pd.to_timedelta(np.random.randint(1,1000000,len(df)),unit='us')
    print(df.head())
    return df

def read_parsedlog(path_to_file):
    """"
     Parse syslog/journalctl into a data frame
    """
    print("Going to read csv",path_to_file)
    colspecs = [(0,27),(27,-1)]
    df =pd.read_fwf(path_to_file,colspecs=colspecs)
    df.columns=["ds","y_org"]
    df["y"] = 0
    #format=Mar 31 09:32:06 
    print(df.head())
    return df

def do_similarty(str1,str2,counter):
    doc1 = nlp(str1)
    doc2 = nlp(str2)
    similarity = doc1.similarity(doc2)
    #print(similarity)
    return similarity,counter

if __name__ == '__main__':
    """
    Analyze a log file and convert converted to Pandas Dataframe and log text changed 
    to relative sentence similarity using Spacy word2vec library
    """

    print(pd.__version__)
    print("""Arguments <path to the log> <logtype>   <output path>
     logtype =1 for syslog
     logtype =2 for Parsed CSV
     as argument""")

    if len(sys.argv) != 4:  
        print("Error Provide the arguments")
        sys.exit()

   # "/home/alex/Downloads/dc4/journal.log"
    
    logfile = sys.argv[1] 
    outfile = sys.argv[3]
    df = None
    if (sys.argv[2] == "1"):
        df = read_syslog(logfile)
    if (sys.argv[2] == "2"):
        df = read_parsedlog(logfile)
    
    nlp = spacy.load("en_core_web_sm")

   # Let's use all the cores
    pool = Pool(processes=16)
    results =[] # Output of multi Processing results
    # Let's check the similarity of one line with the next
    str1 = None
    str2 = None
    for index, row in df.iterrows():
        if str1 == None:
            str1 =row['y_org']
        else:
            str2 = row['y_org']
            results.append(pool.apply_async(do_similarty,(str1,str2,index,)))
            str1 = str2
    pool.close()
    pool.join()


    for result in results:
        similarity,index = result.get()
        df.loc[index, 'y'] = similarity
       
    print("----------After assinging----------")
    print(df.head())
    print("---------Going to Picke the dataframe---------")
    df.to_pickle(outfile)


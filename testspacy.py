from os import system
from typing import Iterator
import spacy
import sys
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import jellyfish
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV

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
        # format=Mar 31 09:32:06
        df["ds"] = pd.to_datetime(df["@timestamp"], format="%B %d, %Y @ %H:%M:%S.%f")
        print(df.head())
        #df.columns = ["ds_org", "y_org"]
        #df["y"] = 0
        #df["ds"] = "2022 " + df["ds_org"]
        #df["ds"] = pd.to_datetime(df["ds"], format="%Y %b %d %H:%M:%S")
        
        df = df.fillna(method="ffill")
        dflist.append(df)
    df = pd.DataFrame()
    df = pd.concat(dflist, ignore_index=True)
    print("Input data size", df.shape[0])
    print(df.head())
    return df

def vectorize_df(df):

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
    return tfidf_vector, df

def vectorize_df_list(list_of_dfs):

    #token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b' - this is default with numbers
    # we are skpiping text with numbers
    tfidf_vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words='english')  
    updated_list_dfs = []
    for df in list_of_dfs:
        tfidf_vector = tfidf_vectorizer.fit_transform(df['y_org'])
        tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=df['ds_org'], columns=tfidf_vectorizer.get_feature_names())
        tfidf_df['y']=tfidf_df.apply(np.max,axis=1)
        df = df.assign(y=tfidf_df["y"].reset_index(drop=True))
        updated_list_dfs.append(df)
    df = pd.DataFrame()
    df = pd.concat(updated_list_dfs, ignore_index=True)
    print("Input data size", df.shape[0])
    #print(df.head())
    return df

def test_tokenizer():
    corpus = [
        """zzz1-p1-zzz kubelet-wrapper[1977]: E0219 21:00:03.163475    1977 kubelet_volumes.go:154] \
            Orphaned pod "1e6da6ca-9def-11eb-b6b3-248a0794fa16" found, \
            but volume paths are still present on disk : There were a total of 3 errors similar to this. Turn up verbosity to see them."""
    ]

    vectorizer =TfidfVectorizer(token_pattern=u'(?ui)\\b[a-z][a-z-_0-9]{4,}\\b',stop_words='english')  
    print('token_pattern:', vectorizer.token_pattern)

    vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())


def bin_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    print("Input Size=",df.shape[0])
    df.y = df.y.round(5)
    print(df.head())
    #df['freq']=0.0
    frequency = df.y.value_counts().rename_axis('y').reset_index(name='counts')
    print(frequency.head())
    df =df.merge(frequency,on='y',how='outer')
    df =df.drop_duplicates(subset=['y'], keep='first')
    print("Output Size=",df.shape[0])
    print(df.y.value_counts())
    print(df.loc[df['y'] == 0.262400])
    df.to_csv("binned.csv")
    #g_df = df.groupby('y').groups.keys()
    #print("Ouput Size=",g_df)
    #print(g_df.head())
    
    # Usig PD Cut
    #df['bin'] = pd.cut(df['y'], [0.1, .2,.3,.4,.5, .6,.7,.8 ,.9])
    #print(df.head())
    #print(df["bin"].value_counts())
    #with pd.option_context('display.max_colwidth', None):
    #    print(df.loc[df['bin'] ==  pd.Interval(0.7,0.8)]['y_org'])
    #df.loc["bin"] = df["bin"].astype("object")
    #df.to_csv("binned.csv")

#@profile
def compare_strings():
    #str1= "Mar 31 09:08:41  The earth is beautiful"
    str1= "Mar 31 09:08:41 error we are the world, we are the "
    str2= "Mar 01 09:08:42 Exception we are the , we are the children"
    print("NLP Similarity=",nlp(str1).similarity(nlp(str2)))
    print("Diff lib similarity",SequenceMatcher(None, str1, str2).ratio()) 
    print("Jellyfish lib similarity",jellyfish.jaro_distance(str1, str2))

    doc1 = nlp(str1).vector
    doc2 = nlp(str2).vector
    print("doc1.vector shape",doc1.shape)
    print("doc2.vector shape",doc2.shape)

    #pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=1))])
    #principalComponent1 = pipeline.fit_transform(doc1.reshape(-1, 1))
    #principalComponent2 = pipeline.fit_transform(doc2.reshape(-1, 1))
    #print("principalComponents",principalComponent1.shape)
    
    figsize=(10, 6)
    fig = plt.figure(facecolor='w', figsize=figsize)
    ax = fig.add_subplot(1,1,1) 
    ax.scatter(  doc1,
                doc2,
                c = 'r',
                s = 50)
    
    plt.show()

def compare_strings(s1,s2):
    """
     Use TFIDF to compare two strings
    """
    tfidf_vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words='english')  
    tfidf_vector_all = tfidf_vectorizer.fit_transform([s1,s2])
    tfidf_df_all = pd.DataFrame(tfidf_vector_all.toarray(), columns=tfidf_vectorizer.get_feature_names())
    print(tfidf_df_all.head())
    similairy =cosine_similarity([tfidf_df_all.iloc[0]],[tfidf_df_all.iloc[1]])
    print("Cosine similaity=",similairy[0][0])
    return similairy[0][0]


def read_syslog_in_parts(filename):
    """"
     Parse syslog/journalctl into a data frame 
    """
    chunksize = 1000
    colspecs = [(0, 16), (16, -1)]
    dflist = []
    print("--Going to read Syslog", filename, pd.__version__)
    with pd.read_fwf(filename, colspecs=colspecs,chunksize=chunksize) as reader:
        for df in reader:
            print("---------------")
            df.columns = ["ds_org", "y_org"]
            df["ds"] = "2022 " + df["ds_org"]
            df["ds"] = pd.to_datetime(df["ds"], format="%Y %b %d %H:%M:%S")
            # add a microsecond to distinguish logs
            #df["ds"] = df["ds"] + pd.to_timedelta(np.random.randint(1, 10000), unit='ms')
            #df["ds"] = df["ds"] +pd.to_timedelta(np.random.randint(1,1000000,len(df)),unit='us')
            df = df.fillna(method="ffill")
            dflist.append(df)
        print("Input Data list size",len(dflist))
        df = pd.DataFrame()
        df = pd.concat(dflist, ignore_index=True)
        print("Input data size", df.shape[0])
        print(df.head())
        return df
def lda_topic_creation(df):
    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                        max_iter=10,               
    # Max learning iterations
                                        learning_method='online',   
                                        random_state=100,          
    # Random state
                                        batch_size=128,            
    # n docs in each learning iter
                                        evaluate_every = -1,       
    # compute perplexity every n iters, default: Don't
                                        n_jobs = -1,               
    # Use all available CPUs
                                        )
    lda_output = lda_model.fit_transform(df)
    print(lda_model)  # Model attributes

if __name__ == '__main__':
    """
      For testing use
      python /home/coding/anomalydetection/python/loganalysis/testspacy.py ./python/resources/test1.log
      Use python3 -m spacy download en_core_web_sm to download sm or md model
    """
    
    s1,s2 ="How does the Surface Pro himself 4 compare with iPad Pro?","Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?"
    s1,s2 ="By scrapping the 500 and 1000 rupee notes, how is RBI planning to fight against issue black money?",\
        "How will the recent move to declare 500 and 1000 denomination lewin illegal will curb black money?"
    s1,s2 = "How can I reduce my belly fat through a diet?","How can I reduce my lower belly fat in one month?"
    diff = compare_strings(s1,s2)
    if diff < .70:
        print("Strings are dissimilar")
    else:
        print("Strings are similar")

    sys.exit(0)
    
    pd.set_option('display.width', 1000)
    df = read_csv(sys.argv[1])
    tfidf_vector,df = vectorize_df(df)
    #a Latent Dirichlet Allocation (LDA) 
    lda_topic_creation(tfidf_vector)
    
    test_tokenizer()
    
    bin_dataframe('/home/alex/coding/anomalydetection/df_rank.csv')
    
    
    #print("NLP vector doc1 -->",doc1.vector_norm)
    #print("NLP vector doc2 -->",doc2.vector_norm)
    
    df_list = read_syslog_in_parts(sys.argv[1]) 
   
    df = vectorize_df_list(df_list)
    print(df)
    
    # Using TFidfVectorizer 
    # https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html
    tfidf_vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words='english')  #token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b'
    
    df = read_syslog(sys.argv[1]) 
    tfidf_vector = tfidf_vectorizer.fit_transform(df['y_org'])
    print(tfidf_vectorizer.get_feature_names_out())
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=df['ds_org'], columns=tfidf_vectorizer.get_feature_names())
    # Create a new row with sum of all the terms of the existing rows
    tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()
    tfidf_df['max']=tfidf_df.apply(np.max,axis=1)
    tfidf_df.to_csv('tfidf_df.csv')
    print("tfidf_df.head()")
    print(tfidf_df.head())

    df = tfidf_df.loc['00_Document Frequency']
    #tfidf_df.sort_values(by=['00_Document Frequency','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
    print("Pandas Series Sort Values",df.sort_values())
    tfidf_df.to_csv('tfidf_df.csv')

    #tokenizer = vectorizer.build_tokenizer()
    #for index,row in df.iterrows():
    #    tokens = tokenizer(row['y_org'])
    #    trsfm = vectorizer.fit_transform(tokens)
    #    print(vectorizer.get_feature_names_out())
    #
    nlp = spacy.load("en_core_web_sm")
    #nlp = spacy.load("en_core_web_md")
    
    
    # test the main flow
    #main()

    """
    
   kernprof -l -v ./python/loganalysis/testspacy.py
    NLP Similarity= 0.9999999821467294
    Diff lib similarity 0.5897435897435898
    Jellyfish lib similarity 0.8561253561253562
    Wrote profile results to testspacy.py.lprof
    Timer unit: 1e-06 s

    Total time: 0.043654 s
    File: ./python/loganalysis/testspacy.py
    Function: main at line 32

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
        32                                           @profile
        33                                           def main():
        34         1          1.0      1.0      0.0      str1= "Mar 31 09:08:41  The world is beautiful"
        35         1          0.0      0.0      0.0      str2= "Mar 31 19:08:42  Beautiful is the world"
        36         1      43248.0  43248.0     99.1      print("NLP Similarity=",nlp(str1).similarity(nlp(str2)))
        37         1        375.0    375.0      0.9      print("Diff lib similarity",SequenceMatcher(None, str1, str2).ratio()) 
        38         1         30.0     30.0      0.1      print("Jellyfish lib similarity",jellyfish.jaro_distance(str1, str2))

    """

    """
    Problem with simplistic Data frame groupby

    dfgroup {'2freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.751333697 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11856 (rc: 32)': [0],
     '3DDDDfreeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.888424524 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11864 (rc: 32)': [1],
     '4freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.783381048 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11857 (rc: 32)': [2],
      '5freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.826483871 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11858 (rc: 32)': [3],
       'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.156622971 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11859 (rc: 32)': [4],
        'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.441979340 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11860 (rc: 32)': [5],
         'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.501112749 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11861 (rc: 32)': [6],
          'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.744908809 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11862 (rc: 32)': [7],
           'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.824780730 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11863 (rc: 32)': [8],
            'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.973196071 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11865 (rc: 32)': [9],
             'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.006932867 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11866 (rc: 32)': [10],
              'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.040462344 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11867 (rc: 32)': [11],
               'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.329064958 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11868 (rc: 32)': [12],
                'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.379726157 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11869 (rc: 32)': [14], 
                'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.447227230 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11870 (rc: 32)': [15],
                 'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.511751956 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11871 (rc: 32)': [16],
                  'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.556709131 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11872 (rc: 32)': [17],
                   'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.649002525 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11873 (rc: 32)': [18],
                    'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.725300878 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11874 (rc: 32)': [19],
                     'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.956672325 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11875 (rc: 32)': [20],
                      'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:50.991657869 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11876 (rc: 32)': [21],
                       'freeipa-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:51.310555319 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11877 (rc: 32)': [22], ....
"""
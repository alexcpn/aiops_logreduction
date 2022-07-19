"""
Trying to use Quora data set to improve similarity between two sentences
"""
# Resources
# /home/alex/Downloads/quora-question-pairs

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html ,dcc
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import train_test_split
import scipy
import xgboost as xgb
import numpy as np

def load_csv(path):
    df = pd.read_csv(path)
    print("input size",df.shape[0])
    print(df.head())
    df = df.fillna(method="ffill")
    return df


def compare_strings(s1,s2):
    """
     Use TFIDF to compare two strings
    """
    print("s1",s1)
    print("s2",s2)
    tfidf_vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b[a-z][a-z-_/0-9]{2,}\\b',stop_words='english')  
    tfidf_vector_all = tfidf_vectorizer.fit([s1,s2])
    tfidf_vector_s1 = tfidf_vector_all.transform([s1])
    tfidf_vector_s2 = tfidf_vector_all.transform([s2])
    print("Features names Length", len(tfidf_vectorizer.get_feature_names()))
    print("tfidf_vector_s1 Length",tfidf_vector_s1.toarray().shape)
    print("tfidf_vector_s2 Length",tfidf_vector_s2.toarray().shape)
    print("Features names\n", tfidf_vectorizer.get_feature_names())
    print("tfidf_vector_s2 toarray()")
    print(tfidf_vector_s2.toarray())
    print("tfidf_vector_s2")
    print(tfidf_vector_s2)
    
    similairy =cosine_similarity(tfidf_vector_s1,tfidf_vector_s2)
    print("Cosine similaity=",similairy[0][0])
    return similairy[0][0]


def tokenize_questions(df):

    #take only 20 records for the time being

    df = df.tail(2)
    df = df.reset_index()

    df['combined'] = df['question1'] + df ['question2']

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

    print("tokenize_questions")
    print(df.head())    

    tfidf_vectorizer = TfidfVectorizer(
        # token_pattern=u'(?ui)\\b[a-z][a-z-_/0-9]{3,8}\\b', stop_words='english')
        # trying without numbers
        token_pattern=u'(?ui)\\b[a-z][a-z-_/]{2,}\\b',stop_words='english')

    tfidf_vector = tfidf_vectorizer.fit(df['combined'])
    tfidf_vector_q1 = tfidf_vectorizer.transform(df['question1'])
    print("tfidf_vector",tfidf_vector_q1)

    tfidf_vector_q2 = tfidf_vectorizer.transform(df['question2'])
    print("tfidf_vector",tfidf_vector_q2)
    print(list(tfidf_vector.vocabulary_.keys()))

  
def test_tokenize(df):
    i =0
    for index, row in df.iterrows():
        compare_strings(row['question1'], row['question2'])
        i+=1
        if i >1:
            break

def test_tokenize2(df):
    # https://towardsdatascience.com/finding-similar-quora-questions-with-bow-tfidf-and-random-forest-c54ad88d1370
    # not able to understand the logic of this  
    tfidf_vect = TfidfVectorizer(
        # token_pattern=u'(?ui)\\b[a-z][a-z-_/0-9]{3,8}\\b', stop_words='english')
        # trying without numbers
        token_pattern=u'(?ui)\\b[a-z][a-z-_/]{3,8}\\b', stop_words='english')
    tfidf_vect.fit(pd.concat((df['question1'],df['question2'])).unique())
    trainq1_trans = tfidf_vect.transform(df['question1'].values)
    trainq2_trans = tfidf_vect.transform(df['question2'].values)
    labels = df['is_duplicate'].values
    X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
    y = labels
    X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

    xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, \
     gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 
    xgb_prediction = xgb_model.predict(X_valid)
    print('word level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))
    print('word level tf-idf validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))
    print(classification_report(y_valid, xgb_prediction))


# Dash app name
app = Dash(__name__)


if __name__ == '__main__':

    # Step 1 load

    path_to_data = "/home/alex/Downloads/quora-question-pairs/train.csv"
    df_train =load_csv(path_to_data)
    df = df_train.groupby("is_duplicate")['id'].count()
    print(df.head())

    fig = make_subplots(rows=1, cols=1)
    fig = px.bar(df)
    
    # Step 2 Vectorize with TFIDF
    test_tokenize(df_train)
    #tokenize_questions(df_train)






    # Below for Dash Display

    app.layout = html.Div(children=[
        html.H1(children='Quora Similairty Tester'),

        html.Div(children='''
            Sentence similarity exploration'''),

        html.Br(),

        html.Div([
            html.Div("Log file Name = {}".format(path_to_data)),
            html.Br(),
                  ], style={'color': 'blue', 'fontSize': 14}),

        html.Br(),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),
        html.Br()
    ])
    app.run_server(debug=True, use_reloader=False)  
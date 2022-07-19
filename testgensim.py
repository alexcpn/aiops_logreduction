"""
Using Gensim to test similarity
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim

from gensim import models


print("Text Similarity with Gensim and Scikit utils")
# compute vector space with sklearn
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# Using Scikit learn feature extractor

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english')
corpus_vect = vect.fit_transform(documents)
# take the dict keys out
texts = list(vect.vocabulary_.keys())

from gensim import corpora
dictionary = corpora.Dictionary([texts])

# transform scikit vocabulary into gensim dictionary
corpus_vect_gensim = gensim.matutils.Sparse2Corpus(corpus_vect, documents_columns=False)

# create LSI model
lsi = models.LsiModel(corpus_vect_gensim, id2word=dictionary, num_topics=9)

# convert the query to LSI space
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
print("vec_bow",vec_bow)
vec_lsi = lsi[vec_bow]  
print(vec_lsi)

# Find similarities
from gensim import similarities
index = similarities.MatrixSimilarity(lsi[corpus_vect_gensim])  # transform corpus to LSI space and index it

sims = index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples

sims = sorted(enumerate(sims), key=lambda item: -item[1])
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])
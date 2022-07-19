import gensim.downloader as api
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

wv = api.load('word2vec-google-news-300')

for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")


try:
    vec_king = wv['king']
    print("vector king",vec_king.shape[0])
    vec_castle = wv['castle']
    print("vector castle",vec_castle.shape[0])

    vec_cameroon = wv['cameroon']
    print("vector cmeroon",vec_cameroon.shape[0])
except KeyError as e:
    print("The word {} does not appear in this model".format(e.args[0]))

pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))


print(wv.most_similar(positive=['car', 'minivan'], topn=5))

print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))
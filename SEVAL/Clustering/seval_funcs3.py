from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from SEVAL.Tools import SpacyFuncs
import collections
import pandas as pd
import scipy.stats
import numpy as np
import re

# ignore divide by zero
np.seterr(divide='ignore', invalid='ignore')


def text_2_list(corpus):
    # Get raw text as string.
    with open(corpus) as f:
        text = f.read()
        f.close()

    documents = SpacyFuncs.break_sentences(text)

    return documents


def cluster_texts(documents, true_k,):

    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(documents)

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(x)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    # Todo: find cluster closeness (sequence similarity)
    terms = vectorizer.get_feature_names()
    l_word = max(terms, key=lambda s: (len(s), s))
    return terms, order_centroids, l_word


def count_words_in_clus(true_k, order_centroids, terms, sentence, word_count, l_word):

    # initialise counters
    clus_list = []
    words_in_clus = []
    # nc_wc = no cluster word count
    nc_wc = 0
    clus_size = 20
    # split into list of words
    word_list = sentence.split()

    hits_2d = np.array([[" " * len(l_word) for x in range(true_k)] for y in range(clus_size)])
    for i in range(true_k):
        count = 0
        for ind in order_centroids[i, : clus_size]:
            # insert into 1D array
            clus_list.insert(i, terms[ind])
            # insert into 2D array
            hits_2d[count, i] = terms[ind]
            count += 1
    hit_list = collections.Counter(clus_list)

    # Transform word_list into probabilities
    prob_list = []
    # loop through clusters
    for i in range(true_k):
        # loop through hits
        for x in range(clus_size):
            # extract word from 2D Array
            word = re.sub("[^a-zA-Z]+", "", (str(hits_2d[x, i])))

            # multiple hits
            if word in word_list and word in hit_list and hit_list[word] > 1:
                prob_list.append(1 / hit_list[word])

            # exactly one hit
            elif word in word_list and word in hit_list and hit_list[word] == 1:
                prob_list.append(1 / word_count)
            # no hits
            else:
                nc_wc += 1
                prob_list.append(0)

    prob_list.append(nc_wc / word_count)

    sum(prob_list)

    ent = entropy(word_list, base=2)

    duo_ent = entropy([len(absolute_hits) / word_count, (word_count - len(absolute_hits)) / word_count], base=2)

    return [len(absolute_hits), duo_ent, ent]


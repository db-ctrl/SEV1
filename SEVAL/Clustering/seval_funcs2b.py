from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from SEVAL.Tools import SpacyFuncs
import collections
import pandas as pd
import scipy.stats
import numpy as np

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
    return terms, order_centroids


def count_words_in_clus(true_k, order_centroids, terms, sentence, word_count):

    # initialise counters
    clus_list = []
    words_in_clus = []
    absolute_hits = []
    # nc_wc = no cluster word count
    nc_wc = 0
    clus_size = 20
    # split into list of words
    word_list = sentence.split()
    # check if a specific word is in a cluster
    for i in range(true_k):
        print("Cluster %d:" % i),
        hits = 0
        # Print x amount of words from each cluster
        for ind in order_centroids[i, : clus_size]:
            clus_list.insert(i, terms[ind])
            print(' %s' % terms[ind])
            # check if a specific word is in a cluster
            if terms[ind] in word_list:
                hits += 1
        words_in_clus.append(hits / word_count)
    hit_list = collections.Counter(clus_list)

    # Transform word_list into probabilities

    for i in range(len(word_list)):

        # Multiple hits
        if word_list[i] in hit_list and hit_list[word_list[i]] > 1:
            word_list[i] = (1 / hit_list[word_list[i]])

        # exactly one hit
        elif word_list[i] in hit_list and hit_list[word_list[i]] == 1:
            word_list[i] = (1 / word_count)

        # no hits
        else:
            nc_wc += 1
            word_list[i] = 0
    sum(word_list)
    word_list.append(nc_wc / word_count)
   # sum(words_in_clus)
    ent = entropy(word_list, base=2)

    duo_ent = entropy([len(absolute_hits) / word_count, (word_count - len(absolute_hits)) / word_count], base=2)

    return [len(absolute_hits), duo_ent, ent]


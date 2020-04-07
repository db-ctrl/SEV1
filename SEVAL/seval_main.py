from SEVAL.Clustering import seval_funcs2
from SEVAL.g_sheets import gs_funcs
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from collections import Counter
###
# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("/Users/david/PycharmProjects/SEVAL/Creds/seval-270420-f0725c9a2188.json", scope)
client = gspread.authorize(creds)

CORPUS_PATH = '/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/HP5.txt'

# Find a workbook by name and open the first sheet
sheet = client.open("AutoSentenceEval").sheet1

sheet_range = len(sheet.col_values(1))
# initialise g_sheet row
row = 2
# choose amount of clusters
true_k = 250


def generate_clusters(true_k):

    # convert raw text into documents
    documents = seval_funcs2.text_2_list(CORPUS_PATH)

    # Generate clusters
    terms, order_centroids, l_word = seval_funcs2.cluster_texts(documents, true_k)

    return terms, order_centroids, l_word


def generate_data(row, true_k, order_centroids, terms, l_word):

    for i in range(sheet_range):

        # Get values from g_sheet
        sentence = gs_funcs.get_bare_sentence(row)

        # calculate cluster metrics
        words_in_clus, duo_ent, entropy, word_count = seval_funcs2.count_words_in_clus(true_k, order_centroids, terms, sentence, l_word)

        # update values in g_sheet
        gs_funcs.update_readability_metrics(row, sentence)
        gs_funcs.update_cluster_metrics(row, words_in_clus, duo_ent, entropy, word_count)

        row += 1

# ---------------------------- Working function ----------------------------


# terms, order_centroids, l_word = generate_clusters(true_k)

# generate_data(row, true_k, order_centroids, terms, l_word)

gs_funcs.normalise_data(row)

# for j in range(sheet_range):
#    gs_funcs.set_auto_scores(row)
#    row += 1
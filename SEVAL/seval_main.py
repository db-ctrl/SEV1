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

# convert raw text into documents
documents = seval_funcs2.text_2_list(CORPUS_PATH)

# initialise g_sheet row
row = 2
cols_to_norm = [6]
# choose amount of clusters

true_k = 250

# Generate clusters

terms, order_centroids, l_word = seval_funcs2.cluster_texts(documents, true_k)

for i in range(len(sheet.col_values(1))):
    # ignore blank values
    if len(sheet.cell(row, 2).value.split()) == 0:
        pass
    # Get values from g_sheet
    sentence = gs_funcs.get_bare_sentence(row)

    # calculate cluster metrics
    words_in_clus, duo_ent, entropy, word_count = seval_funcs2.count_words_in_clus(true_k, order_centroids, terms, sentence, l_word)

    # update values in g_sheet
    gs_funcs.update_readability_metrics(row, sentence)
    gs_funcs.update_cluster_metrics(row, words_in_clus, duo_ent, entropy, word_count)

    # normalise data
    gs_funcs.normalise_data()

row += 1




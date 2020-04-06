
import textstat
import time
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import math
import re
import numpy as np

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name("/Users/david/PycharmProjects/SEVAL/Creds/seval-270420-f0725c9a2188.json", scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
sheet = client.open("AutoSentenceEval").sheet1


def update_readability_metrics(row, sentence):

    # Update Word Count
    sheet.update_cell(row, 6, (len(sentence.split()))),
    time.sleep(1)
    # Update Flesch Reading Ease
    sheet.update_cell(row, 7, (textstat.flesch_reading_ease(sheet.cell(row, 2).value))),
    time.sleep(1)
    # Update Gunning Fog Index
    sheet.update_cell(row, 8, (textstat.gunning_fog(sheet.cell(row, 2).value))),
    time.sleep(1)


def update_cluster_metrics(row, words_in_clus, duo_ent, ent, word_count):

    # Update words in cluster
    sheet.update_cell(row, 9, str(words_in_clus / word_count)),
    time.sleep(1)

    # Update entropy
    sheet.update_cell(row, 10, duo_ent),
    time.sleep(1)
    sheet.update_cell(row, 11, ent),
    time.sleep(1)


def normalise_data(row):

    # specify columns to normalise
    columns = [7, 8, 17]

    for col in columns:

        # set row
        row = 2

        metrics = sheet.col_values(col)

        # remove header
        metrics = metrics[1:]

        # convert to floats
        for i in range(len(metrics)):
            if metrics[i] == '':
                metrics[i] = 0.0
            else:
                metrics[i] = float(metrics[i])

        # TODO: amend norm calculation to generate proper values (max should be 1, min should be 0)
        normal_data = metrics / (np.linalg.norm(metrics))
        for value in normal_data:
            sheet.update_cell(row, col, value)
            row += 1
            time.sleep(1)


def get_bare_sentence(row):

    sentence = sheet.cell(row, 2).value
    # convert sentence to lowercase
    sentence = sentence.lower()
    # remove ignored characters from text
    sentence = sentence.replace('\'', '')
    sentence = sentence.replace('\n', ' ')
    # remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)

    return sentence


def auto_score1(row):

    auto_score = ''

    return auto_score

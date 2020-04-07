
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
    columns = [13, 14, 15, 16]

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

        # normal_data = metrics / (np.linalg.norm(metrics))

        normal_data = [float(i)/max(metrics) for i in metrics]

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


def get_norm_values(row):

    fre = float(sheet.cell(row, 7).value)
    gfi = float(sheet.cell(row, 8).value)
    wic = float(sheet.cell(row, 9).value)
    ent1 = float(sheet.cell(row, 10).value)
    ent2 = float(sheet.cell(row, 11).value)

    return fre, gfi, wic, ent1, ent2


def set_auto_scores(row):

    fre, gfi, wic, ent1, ent2 = get_norm_values(row)

    # calculate autoscores
    as1 = ((1.5 * fre) + (1.1 * gfi) + (0.6 * wic) + math.pow((1.8 * ent1), ent2)) / 5
    as2 = (math.pow(fre, gfi) + (1 / wic) + math.sin(ent2 + math.cos(ent1))) / 5
    as3 = (math.pow(wic, gfi) + (1 / fre) + (ent1 / ent2)) / 5
    as4 = (((math.pow(math.tan(wic), ent2)) / (math.pow(math.sin(ent1), fre))) + (0.6 * gfi) / 5)

    # update autoscores
    sheet.update_cell(row, 13, as1),
    time.sleep(1)

    sheet.update_cell(row, 14, as2),
    time.sleep(1)

    sheet.update_cell(row, 15, as3),
    time.sleep(1)

    sheet.update_cell(row, 16, as4),
    time.sleep(1)






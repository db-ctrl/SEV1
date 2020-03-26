
from collections import Counter
import textstat
import time
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import math
import re

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name("/Users/david/PycharmProjects/SEVAL/Creds/seval-270420-f0725c9a2188.json", scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
sheet = client.open("AutoSentenceEval").sheet1


def update_readability_metrics(row):

    # Update Word Count
    sheet.update_cell(row, 6, (len(sheet.cell(row, 2).value.split()))),
    time.sleep(1)
    # Update Flesch Reading Ease
    sheet.update_cell(row, 7, (textstat.flesch_reading_ease(sheet.cell(row, 2).value))),
    time.sleep(1)
    # Update Gunning Fog Index
    sheet.update_cell(row, 8, (textstat.gunning_fog(sheet.cell(row, 2).value))),
    time.sleep(1)


def update_cluster_metrics(row, words_in_clus, duo_ent, ent):
    # check if row empty
    if len(sheet.cell(row, 2).value.split()) == 0:
        pass
    else:
        # Update words in cluster
        sheet.update_cell(row, 9, str(words_in_clus) + "/" + str(len(sheet.cell(row, 2).value.split()))),
        time.sleep(1)
        # Update entropy
        if math.isnan(duo_ent):
            sheet.update_cell(row, 10, "N/A"),
        else:
            sheet.update_cell(row, 10, duo_ent),
            sheet.update_cell(row, 11, ent),
        time.sleep(1)


def get_word_count(row):
    word_c = len(sheet.cell(row, 2).value.split())
    return word_c


def get_sentence(row):

    sentence = sheet.cell(row, 2).value
    # remove ignored characters from text
    sentence = sentence.replace('\'', '')
    sentence = sentence.replace('\n', ' ')

    return sentence


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


def update_entropy(row, entropy, duo_ent):

    if len(sheet.cell(row, 2).value.split()) == 0:
        pass
    else:
        # Update entropy
        if math.isnan(entropy):
            sheet.update_cell(row, 10, "N/A"),
        else:
            sheet.update_cell(row, 10, duo_ent),
            sheet.update_cell(row, 11, entropy),

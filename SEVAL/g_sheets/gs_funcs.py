
from collections import Counter
import textstat
import time
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import math

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name("/Users/david/PycharmProjects/SEVAL/Creds/seval-270420-f0725c9a2188.json", scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
sheet = client.open("AutoSentenceEval").sheet1


def update_metrics(row, words_in_clus, entropy):
    # check if row empty
    if len(sheet.cell(row, 2).value.split()) == 0:
        pass
    else:
        # Update Word Count
        sheet.update_cell(row, 6, (len(sheet.cell(row, 2).value.split()))),
        time.sleep(1)
        # Update Flesch Reading Ease
        sheet.update_cell(row, 7, (textstat.flesch_reading_ease(sheet.cell(row, 2).value))),
        time.sleep(1)
        # Update Gunning Fog Index
        sheet.update_cell(row, 8, (textstat.gunning_fog(sheet.cell(row, 2).value))),
        time.sleep(1)
        # Update words in cluster
        sheet.update_cell(row, 9, str(words_in_clus) + "/" + str(len(sheet.cell(row, 2).value.split()))),
        time.sleep(1)
        # Update entropy
        if math.isnan(entropy):
            sheet.update_cell(row, 10, "N/A"),
        else:
            sheet.update_cell(row, 10, entropy),
        time.sleep(1)


def get_word_count(row):
    word_c = len(sheet.cell(row, 2).value.split())
    return word_c


def get_sentence(row):
    sentence = sheet.cell(row, 2).value
    # convert sentence to lowercase
    sentence.lower()
    return sentence


def update_entropy(row, entropy):

    if len(sheet.cell(row, 2).value.split()) == 0:
        pass
    else:
        # Update entropy
        if math.isnan(entropy):
            sheet.update_cell(row, 10, "N/A"),
        else:
            sheet.update_cell(row, 10, entropy),

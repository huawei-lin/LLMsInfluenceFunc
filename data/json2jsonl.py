import os, json
import pandas as pd


json_list = []
with open('../alpaca_data.json') as json_file:
    data = json.load(json_file)
    json_list = data

#Now, output the list of json data into a single jsonl file
with open('normal_alpaca_data.jsonl', 'w') as outfile:
    for entry in json_list:
        json.dump(entry, outfile)
        outfile.write('\n')

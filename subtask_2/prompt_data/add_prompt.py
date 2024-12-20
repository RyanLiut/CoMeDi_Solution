import pandas as pd
import sys
import csv
import os

def update_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t')
    for i, row in df.iterrows():
        original_context = row['context']
        lemma = row['lemma']

        new_context = f'In this sentence "{original_context}", "{lemma}" means in one word :'

        colon_index = new_context.rfind(':')
        a, b = colon_index, colon_index + 1 

        df.at[i, 'context'] = new_context
        df.at[i, 'indices_target_token'] = f'{a}:{b}'

    df.to_csv(file_path, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"File '{file_path}' has been updated successfully.")

if __name__ == "__main__":
    ROOT_DIR = ""
    dir_path = ROOT_DIR + "/subtask_1/prompt_data"
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            if file_path[-8:] == "uses.tsv":
                update_tsv(file_path)

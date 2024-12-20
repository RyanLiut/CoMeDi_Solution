'''
To utilize the results of subtask 1, get a score from several models
'''

import sys
import os
import os.path
import csv
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import json

def average_absolute_difference(lst):
    pairs = list(itertools.combinations(lst, 2))
    
    absolute_differences = [abs(a - b) for a, b in pairs]
    
    if absolute_differences:
        average_difference = sum(absolute_differences) / len(absolute_differences)
        return average_difference
    else:
        return 0
    
def get_variation_ratio(lst):
    lst_mode = Counter(lst).most_common(1)[0][0]
    n_diff = sum([i != lst_mode for i in lst])
    return n_diff / len(lst)

def get_variance(lst):
    return np.std(lst)


if __name__ == "__main__":
    # as per the metadata file, input and output directories are the arguments
    ROOT_DIR = "/XXX/CoMeDi_Solution"
    out_dir = ROOT_DIR + "/subtask_2/thr_based/answer_ensemble/"
    if not os.path.exists(out_dir):      
        os.makedirs(out_dir)

    languages = ['chinese', 'english', 'german', 'norwegian', 'russian', 'spanish', 'swedish']
    columns = ['SCORE_ALL', 'SCORE_CHINESE', 'SCORE_ENGLISH', 'SCORE_GERMAN', 'SCORE_NORWEGIAN', 'SCORE_RUSSIAN', 'SCORE_SPANISH', 'SCORE_SWEDISH']

    language2column = {'average': 'SCORE_AVERAGE', 'chinese': 'SCORE_CHINESE','english': 'SCORE_ENGLISH', 'german': 'SCORE_GERMAN', 'norwegian': 'SCORE_NORWEGIAN', 'russian': 'SCORE_RUSSIAN', 'spanish': 'SCORE_SPANISH', 'swedish': 'SCORE_SWEDISH'}

    # Multiple results to ensemble
    submission_path_list = [
        "/ANSWER_DIR_0",
        "/ANSWER_DIR_1",
        "/ANSWER_DIR_2",
    ]
    metric = "STD" # Other choices: "MDP", "VR"
    for language in languages:
        # Load submission file
        submission = {}

        for submission_dir in submission_path_list:
            submission_path = submission_dir +"/" + language + '.tsv'
            submission_file = open(submission_path, mode='r')
            reader = csv.DictReader(submission_file, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
            
            for row in reader:
                key = (row['identifier1'], row['identifier2'])
                if not key in submission:
                    submission[key] = []
                submission[key].append(float(row['prediction_sim']))

        for k,v in submission.items():
            if metric == "STD":
                submission[k] = np.std(v)
            elif metric == "VR":
                submission[k] = get_variance(v)
            elif metric == "MDP":
                submission[k] = average_absolute_difference(v)
            else:
                print("Error: metric is out of choices")
                sys.exit()
            

        df_temp = pd.DataFrame()
        df_temp['identifier1'] = [k[0] for k,v in submission.items()]
        df_temp['identifier2'] = [k[1] for k,v in submission.items()]
        df_temp['prediction'] = [v for k,v in submission.items()]
        df_temp.to_csv(out_dir +language +'.tsv',index = False, sep='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')
    print(out_dir)
    
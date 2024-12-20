import sys
import os
import os.path
import csv
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('krippendorff')
import krippendorff
import numpy as np

# as per the metadata file, input and output directories are the arguments
[_, input_dir, truth_dir] = sys.argv


languages = ['chinese', 'german', 'english', 'norwegian', 'russian', 'spanish', 'swedish']
columns = ['SCORE_ALL', 'SCORE_CHINESE', 'SCORE_ENGLISH', 'SCORE_GERMAN', 'SCORE_NORWEGIAN', 'SCORE_RUSSIAN', 'SCORE_SPANISH', 'SCORE_SWEDISH']

language2column = {'average': 'SCORE_AVERAGE', 'chinese': 'SCORE_CHINESE','english': 'SCORE_ENGLISH', 'german': 'SCORE_GERMAN', 'norwegian': 'SCORE_NORWEGIAN', 'russian': 'SCORE_RUSSIAN', 'spanish': 'SCORE_SPANISH', 'swedish': 'SCORE_SWEDISH'}

# languages = ['chinese', 'english']
# columns = ['SCORE_ALL', 'SCORE_CHINESE', 'SCORE_ENGLISH']
# language2column = {'average': 'SCORE_AVERAGE', 'chinese': 'SCORE_CHINESE','english': 'SCORE_ENGLISH'}

scores = {}

for language in languages:
    # Load submission file
    submission_file_name = language + '.tsv'
    submission_dir = input_dir #os.path.join(input_dir, 'res')
    submission_path = os.path.join(input_dir, submission_file_name)
    if not os.path.exists(submission_path):
        message = "Error: Expected submission file '{0}', found files {1}"
        sys.exit(message.format(submission_file_name, os.listdir(submission_dir)))

    submission = {}
    with open(submission_path, mode='r') as submission_file:
        reader = csv.DictReader(submission_file, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        for row in reader:
            key = (row['identifier1'], row['identifier2'])
            submission[key] = float(row['prediction'])
    
    # Load truth file
    truth_file_name = language + '/labels.tsv'
    truth_path = os.path.join(truth_dir, truth_file_name)
    if not os.path.exists(truth_path):
        message = "Error: Expected truth file '{0}', found files {1}"
        sys.exit(message.format(truth_file_name, os.listdir(truth_dir)))

    truth = {}
    with open(truth_path, mode='r') as truth_file:
        reader = csv.DictReader(truth_file, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        for row in reader:
            key = (row['identifier1'], row['identifier2'])
            truth[key] = float(row['median_cleaned'])

    # Check submission format
    if set(submission.keys()) != set(truth.keys()) or len(submission.keys()) != len(truth.keys()):
        message = "Error in '{0}': Submitted targets do not match gold targets."
        sys.exit(message.format(truth_path))

    if any((not (i == 1.0 or i == 2.0 or i == 3.0 or i == 4.0) for i in truth.values())):
        message = "Error in '{0}': Submitted values contain values that are not equal to ordinal label range."
        sys.exit(message.format(truth_path))

    # Get submitted values and true values
    submission_values = [submission[target] for target in truth.keys()]
    truth_values = [truth[target] for target in truth.keys()]

    # Calculate score
    data = [truth_values, submission_values]
    score = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    scores[language] = score

# Calculate the average score
average_score = np.mean([scores[language] for language in languages])
scores['average'] = average_score

df = pd.DataFrame([scores])
df = df[['average', 'chinese', 'english', 'german', 'norwegian', 'russian', 'spanish', 'swedish']]
df = df.apply(lambda x: round(x, 4))
df.to_csv(input_dir + "/result.csv", index=False)
print(input_dir)
print(df)


# Write output scores
# with open(os.path.join(input_dir, 'scores.txt'), 'a') as output_file:
#     for language in languages + ['average']:
#         column = language2column[language]
#         score = scores[language]
#         output_file.write("{0}:{1}\n".format(column, score))

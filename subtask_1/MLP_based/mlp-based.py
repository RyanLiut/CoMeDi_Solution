import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile
import torch
from transformers import AutoTokenizer, XLMRobertaModel
from transformers import BertTokenizer, BertModel
from transformers import get_scheduler
from tqdm import tqdm
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import sys
import subprocess
import krippendorff

# check if GPU is available
is_available = torch.cuda.is_available()
print(f"GPU : {is_available}")
device = torch.device("cuda" if is_available else "cpu")


# raw data
path_dev = 'dev/'
path_train = 'train/'
path_test = 'test/'

if not os.path.exists(path_dev):
    os.makedirs(path_dev)
with ZipFile('dev.zip', 'r') as dev:
    dev.extractall(path_dev)
if not os.path.exists(path_train):
    os.makedirs(path_train)
with ZipFile('train.zip', 'r') as train:
    train.extractall(path_train)
if not os.path.exists(path_test):
    os.makedirs(path_test)
with ZipFile('test.zip', 'r') as test:
    test.extractall(path_test)

languages = os.listdir(path_train)

label_file_paths_train = []
uses_file_paths_train = []
instance_file_paths_dev = []
uses_file_paths_dev = []
instance_file_paths_test = []
uses_file_paths_test = []

for lang in languages:
    label_file_paths_train.append(path_train + lang + '/labels.tsv')
    uses_file_paths_train.append(path_train + lang + '/uses.tsv')
    instance_file_paths_dev.append(path_dev + lang + '/instances.tsv')
    uses_file_paths_dev.append(path_dev + lang + '/uses.tsv')
    instance_file_paths_test.append(path_test + lang + '/instances.tsv')
    uses_file_paths_test.append(path_test + lang + '/uses.tsv')


# loading train labels and uses and dev instances and uses (test instances and uses)
# dictionary containing input file paths
paths = {'train_labels_list': label_file_paths_train, 'train_uses_list': uses_file_paths_train,
         'dev_uses_list': uses_file_paths_dev, 'dev_instances_list': instance_file_paths_dev,
         'test_uses_list': uses_file_paths_test, 'test_instances_list': instance_file_paths_test}
# dictionary to store the extracted data
data_dict = {'train_labels_list': [], 'train_uses_list': [],
             'dev_uses_list': [], 'dev_instances_list': [],
             'test_uses_list': [], 'test_instances_list': []}

for save_path, path_list in paths.items():
    for path in path_list:
        with open(path, encoding='utf-8') as tsvfile:
            language = path.split('/')[1]
            reader = csv.DictReader(tsvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')
            for row in reader:
                row['language'] = language
                data_dict[save_path].append(row)

train_labels_list = data_dict['train_labels_list']
train_uses_list = data_dict['train_uses_list']
dev_uses_list = data_dict['dev_uses_list']
dev_instances_list = data_dict['dev_instances_list']
test_uses_list = data_dict['test_uses_list']
test_instances_list = data_dict['test_instances_list']


# make dictionaries to map identifiers to their contexts and target token indices from the train and dev uses data (test uses data)
def create_mappings(uses_list):
    id2context = {}
    id2idx = {}
    for row in uses_list:
        identifier = row['identifier']
        context = row['context']
        idx = row['indices_target_token']
        id2context[identifier] = context
        id2idx[identifier] = idx
    return id2context, id2idx

train_id2context, train_id2idx = create_mappings(train_uses_list)
dev_id2context, dev_id2idx = create_mappings(dev_uses_list)
test_id2context, test_id2idx = create_mappings(test_uses_list)


# merging train labels and uses into a single dataframe
train_uses_merged = []
for row in train_labels_list:
    identifier1_train = row['identifier1']
    identifier2_train = row['identifier2']

    # use id2context dictionary to get the corresponding context for each identifier
    context1 = train_id2context.get(identifier1_train)
    context2 = train_id2context.get(identifier2_train)

    # use id2idx dictionary to get the corresponding target token index for each identifier
    index_target_token1 = train_id2idx.get(identifier1_train)
    index_target_token2 = train_id2idx.get(identifier2_train)

    lemma = row['lemma']
    median_cleaned = row['median_cleaned']
    judgments = row['judgments']
    language = row['language']
    data_row = {'context1': context1, 'context2': context2, 'index_target_token1': index_target_token1,
                'index_target_token2': index_target_token2, 'identifier1': identifier1_train,
                'identifier2': identifier2_train, 'lemma': lemma, 'median_cleaned': median_cleaned,
                'judgments': judgments, 'language': language}

    train_uses_merged.append(data_row)

df_train_uses_merged = pd.DataFrame(train_uses_merged)


# merging dev instances and uses into a single dataframe
dev_uses_merged = []
for row in dev_instances_list:
    identifier1_dev = row['identifier1']
    identifier2_dev = row['identifier2']

    # use id2context dictionary to get the corresponding context for each identifier
    context1 = dev_id2context.get(identifier1_dev)
    context2 = dev_id2context.get(identifier2_dev)

    # use id2idx dictionary to get the corresponding target token index for each identifier
    index_target_token1 = dev_id2idx.get(identifier1_dev)
    index_target_token2 = dev_id2idx.get(identifier2_dev)

    lemma = row['lemma']

    language = row['language']
    data_row = {'context1': context1, 'context2': context2, 'index_target_token1': index_target_token1,
                'index_target_token2': index_target_token2, 'identifier1': identifier1_dev, 'median_cleaned': median_cleaned,
                'identifier2': identifier2_dev, 'lemma': lemma, 'language': language}

    dev_uses_merged.append(data_row)

df_dev_uses_merged = pd.DataFrame(dev_uses_merged)


# merging test instances and uses into a single dataframe
test_uses_merged = []
for row in test_instances_list:
    identifier1_test = row['identifier1']
    identifier2_test = row['identifier2']

    # use id2context dictionary to get the corresponding context for each identifier
    context1 = test_id2context.get(identifier1_test)
    context2 = test_id2context.get(identifier2_test)

    # use id2idx dictionary to get the corresponding target token index for each identifier
    index_target_token1 = test_id2idx.get(identifier1_test)
    index_target_token2 = test_id2idx.get(identifier2_test)

    lemma = row['lemma']

    language = row['language']
    data_row = {'context1': context1, 'context2': context2, 'index_target_token1': index_target_token1,
                'index_target_token2': index_target_token2, 'identifier1': identifier1_test, 'median_cleaned': median_cleaned,
                'identifier2': identifier2_test, 'lemma': lemma, 'language': language}

    test_uses_merged.append(data_row)

df_test_uses_merged = pd.DataFrame(test_uses_merged)


# define and load the tokenizer and model for XLM-RoBERTa
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)

# define and load the tokenizer and model for mBERT
# tokenizer = AutoTokenizer.from_pretrained("bert-base-mult")
# model = BertModel.from_pretrained("bert-base-mult").to(device)


# calculate truncation indices for a sequence of tokens to ensure that the target subwords are preserved
# print('calculate truncation indices for a sequence of tokens to ensure that the target subwords are preserved')
def truncation_indices(target_subword_indices: list[bool], truncation_tokens_before_target=0.5) -> tuple[int, int]:
    max_tokens = 512
    n_target_subtokens = target_subword_indices.count(True)
    tokens_before = int((max_tokens - n_target_subtokens) * truncation_tokens_before_target)
    tokens_after = max_tokens - tokens_before - n_target_subtokens

    # get index of the first target subword
    lindex_target = target_subword_indices.index(True)
    # get index of the last target subword
    rindex_target = lindex_target + n_target_subtokens
    # starting index for truncation
    lindex = max(lindex_target - tokens_before, 0)
    # ending index for truncation
    rindex = rindex_target + tokens_after
    return lindex, rindex

def get_target_token_embedding(context, index, truncation_tokens_before_target=0.5):
    start_idx = int(str(index).strip().split(':')[0])
    end_idx = int(str(index).strip().split(':')[1])

    # tokenize the context with offset mapping
    inputs = tokenizer(context, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False).to(device)

    # offset mapping to provide the start and end positions of each token in the original context
    offset_mapping = inputs['offset_mapping'][0].tolist()

    # convert input ids to tokens
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # create a boolean mask for subwords within the target words span
    subwords_bool_mask = [
        (start <= start_idx < end) or (start < end_idx <= end) or (start_idx <= start and end <= end_idx)
        for start, end in offset_mapping
    ]

    target_token_indices = [i for i, value in enumerate(subwords_bool_mask) if value]

    if not target_token_indices:
        print(
            f"Error: Target token indices not found within the specified range for context: '{context}' and index: '{index}'")
        return None

    # truncate input if it exceeds 512 tokens
    if len(input_ids[0]) > 512:
        # truncation indices based on the subwords boolean mask
        lindex, rindex = truncation_indices(subwords_bool_mask, truncation_tokens_before_target)

        # truncate the tokens, input_ids and subwords_bool_mask within the range of truncation indices
        tokens = tokens[lindex:rindex]
        input_ids = input_ids[:, lindex:rindex]
        subwords_bool_mask = subwords_bool_mask[lindex:rindex]
        offset_mapping = offset_mapping[lindex:rindex]
        inputs['input_ids'] = input_ids  # update the input_ids in the inputs dictionary

        # check if truncation was successful
        if len(input_ids[0]) > 512:
            print(
                f"Truncation failed: input seq len ({len(input_ids[0])}) exceeds the maximum token limit for context: '{context}' and index: '{index}'")
            return None

    # extract the subwords for the target word
    extracted_subwords = [tokens[i] for i, value in enumerate(subwords_bool_mask) if value]

    if not extracted_subwords:
        print(f"Error: no subwords extracted for the target word in context: '{context}' and index: '{index}'")
        return None

    with torch.no_grad():
        outputs = model(inputs['input_ids'], output_hidden_states=True).hidden_states # get embeddings for the truncated input
        # you can freely choose the layers of the model here
        outputs = outputs[-1]

    # embeddings for all tokens in the truncated input
    embeddings = outputs[0]

    # embeddings for target token
    target_embeddings = embeddings[subwords_bool_mask]

    if target_embeddings.size(0) == 0:
        print(f"error: no embeddings found for the target token in context: '{context}' and index: '{index}'")
        return None

    # aggregated target token embedding
    target_embeddings_nump = target_embeddings.mean(dim=0).cpu().numpy()

    return target_embeddings_nump


dataframes = [df_train_uses_merged, df_dev_uses_merged, df_test_uses_merged]
file_names = ['subtask1_train_embeddings.npz', 'subtask1_dev_embeddings.npz', 'subtask1_test_embeddings.npz']

# getting target token embeddings for contexts in train and dev
for df, file_name in zip(dataframes, file_names):
    id2embedding = {}

    for _, row in tqdm(df.iterrows()):
        identifier1 = row['identifier1']
        identifier2 = row['identifier2']

        if identifier1 not in id2embedding:
            embedding1 = get_target_token_embedding(row['context1'], row['index_target_token1'])
            id2embedding[identifier1] = embedding1

        if identifier2 not in id2embedding:
            embedding2 = get_target_token_embedding(row['context2'], row['index_target_token2'])
            id2embedding[identifier2] = embedding2

    # store embeddings in a .npz file using identifiers as keys
    np.savez(file_name, **id2embedding)

dataframes = [df_train_uses_merged, df_dev_uses_merged, df_test_uses_merged]
file_names = ['subtask1_train_embeddings.npz', 'subtask1_dev_embeddings.npz', 'subtask1_test_embeddings.npz']


# organize training set
train_input_list = []
train_language_list = []
train_label_list = []
train_ident1_list = []
train_ident2_list = []
loaded_train_embeddings = np.load('subtask1_train_embeddings.npz')
for _, row in tqdm(df_train_uses_merged.iterrows()):
    context_embedding1 = loaded_train_embeddings[row['identifier1']]
    context_embedding2 = loaded_train_embeddings[row['identifier2']]
    context_embedding1 = context_embedding1.reshape(1, 768)
    context_embedding2 = context_embedding2.reshape(1, 768)
    combined_matrix = np.concatenate((context_embedding1, context_embedding2), axis=1)
    train_input_list.append(combined_matrix.tolist())
    train_language_list.append(row['language'])
    train_label_list.append(int(row['median_cleaned'][0])-1)
    train_ident1_list.append(row['identifier1'])
    train_ident2_list.append(row['identifier2'])

# organize dev set
dev_input_list = []
dev_language_list = []
dev_label_list = []
dev_ident1_list = []
dev_ident2_list = []
loaded_dev_embeddings = np.load('subtask1_dev_embeddings.npz')
for _, row in tqdm(df_dev_uses_merged.iterrows()):
    context_embedding1 = loaded_dev_embeddings[row['identifier1']]
    context_embedding2 = loaded_dev_embeddings[row['identifier2']]
    context_embedding1 = context_embedding1.reshape(1, 768)
    context_embedding2 = context_embedding2.reshape(1, 768)
    combined_matrix = np.concatenate((context_embedding1, context_embedding2), axis=1)
    dev_input_list.append(combined_matrix.tolist())
    dev_language_list.append(row['language'])
    dev_label_list.append(int(row['median_cleaned'][0])-1)
    dev_ident1_list.append(row['identifier1'])
    dev_ident2_list.append(row['identifier2'])

# organize test set
test_input_list = []
test_language_list = []
test_label_list = []
test_ident1_list = []
test_ident2_list = []
loaded_test_embeddings = np.load('subtask1_test_embeddings.npz')
for _, row in tqdm(df_test_uses_merged.iterrows()):
    context_embedding1 = loaded_test_embeddings[row['identifier1']]
    context_embedding2 = loaded_test_embeddings[row['identifier2']]
    context_embedding1 = context_embedding1.reshape(1, 768)
    context_embedding2 = context_embedding2.reshape(1, 768)
    combined_matrix = np.concatenate((context_embedding1, context_embedding2), axis=1)
    test_input_list.append(combined_matrix.tolist())
    test_language_list.append(row['language'])
    test_label_list.append(int(row['median_cleaned'][0])-1)
    test_ident1_list.append(row['identifier1'])
    test_ident2_list.append(row['identifier2'])


# some processing of the dataset before putting it into the model
def trans2TheDataset(data):
    inputs = list()
    languages = list()
    labels = list()
    ident_all1 = list()
    ident_all2 = list()
    for input, language, label, ident1, ident2 in data:
        inputs.append(input)
        languages.append(language)
        labels.append(label)
        ident_all1.append(ident1)
        ident_all2.append(ident2)
    inputs = torch.tensor(inputs).to(device)
    labels = torch.LongTensor(labels).to(device)
    return inputs, languages, labels, ident_all1, ident_all2


# set random number seed
def seedConfig(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TextDataset(Dataset):
    def __init__(self, all_input, all_language, all_label, all_ident1, all_ident2):
        self.all_input = all_input
        self.all_language = all_language
        self.all_label = all_label
        self.all_ident1 = all_ident1
        self.all_ident2 = all_ident2

    def __getitem__(self, index):
        input = self.all_input[index]
        language = self.all_language[index]
        label = self.all_label[index]
        ident1 = self.all_ident1[index]
        ident2 = self.all_ident2[index]
        return input, language, label, ident1, ident2

    def __len__(self):
        return len(self.all_input)


# define MLP model
class MModel(nn.Module):
    def __init__(self, num_labels):
        super(MModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.trans = nn.Linear(768 * 2, 768) # linear layers used to reduce dimensionality
        self.classifier = nn.Linear(768, num_labels) # linear layer used for mapping to classification labels

    def forward(self, inputs):
        hidden_vector = self.trans(inputs.squeeze(1))
        hidden_vector = torch.relu(hidden_vector)
        hidden_vector = self.dropout(hidden_vector)
        logits = self.classifier(hidden_vector) 
        return logits


# training process
def train_loop(train_dataloader, model, loss_fn, optimizer, epoch):
    total_loss_train = 0
    correct = 0
    size = len(train_dataloader.dataset)
    model.train()
    for inputs, languages, labels, ident1, ident2 in tqdm(train_dataloader, colour='green'):

        logits = model(inputs)
        loss = loss_fn(logits, labels)
        total_loss_train += loss.item()

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        lr_scheduler.step()  
        correct += (logits.argmax(1) == labels).type(torch.float).sum().item()
    correct /= size
    print(f'Epoch {epoch} Train Acc: {(100 * correct):>0.6f}%   Loss: {total_loss_train / size:.5f}', flush=True)
    return total_loss_train


# dev process (also test process)
def valid_loop(dataloader, model, epoch, record):
    size = len(dataloader.dataset)
    correct = 0
    model.eval()
    for inputs, languages, labels, ident1, ident2 in tqdm(dataloader, colour='blue'):
        logits = model(inputs)
        logits_softmax = torch.softmax(logits, dim=1).tolist()
        pre_list = logits.argmax(1).tolist()
        pre_list_median = []
        for pre in pre_list:
            pre_list_median.append(str(pre + 1) + '.0')
        for ide1, ide2, language, pre, pre_logits in zip(ident1, ident2, languages, pre_list_median, logits_softmax):
            record[language]['ident1'].append(ide1)
            record[language]['ident2'].append(ide2)
            record[language]['prediction'].append(pre)
            record[language]['prob_max'].append(max(pre_logits))
            record[language]['prob_0'].append(pre_logits[0])
            record[language]['prob_1'].append(pre_logits[1])
            record[language]['prob_2'].append(pre_logits[2])
            record[language]['prob_3'].append(pre_logits[3])

        correct += (logits.argmax(1) == labels).type(torch.float).sum().item()
    correct /= size
    print(f'Epoch {epoch} Valid Acc: {(100 * correct):>0.6f}% ', flush=True)
    return correct


def get_krippendorff(path_pre: str, path_true: str):
    # as per the metadata file, input and output directories are the arguments  
    [_, input_dir1, input_dir2] = sys.argv, path_pre, path_true

    languages = ['chinese', 'german', 'english', 'norwegian', 'russian', 'spanish', 'swedish']
    columns = ['SCORE_ALL', 'SCORE_CHINESE', 'SCORE_ENGLISH', 'SCORE_GERMAN', 'SCORE_NORWEGIAN', 'SCORE_RUSSIAN',
               'SCORE_SPANISH', 'SCORE_SWEDISH']

    language2column = {'average': 'SCORE_AVERAGE', 'chinese': 'SCORE_CHINESE', 'english': 'SCORE_ENGLISH',
                       'german': 'SCORE_GERMAN', 'norwegian': 'SCORE_NORWEGIAN', 'russian': 'SCORE_RUSSIAN',
                       'spanish': 'SCORE_SPANISH', 'swedish': 'SCORE_SWEDISH'}

    scores = {}
    for language in languages:
        # Load submission file
        submission_file_name = language + '.tsv'
        # submission_dir = os.path.join(input_dir, 'res')
        # submission_path = os.path.join(submission_dir, submission_file_name)
        submission_path = os.path.join(input_dir1, submission_file_name)
        # if not os.path.exists(submission_path):
        #     message = "Error: Expected submission file '{0}', found files {1}"
        #     sys.exit(message.format(submission_file_name, os.listdir(submission_dir)))

        submission = {}
        with open(submission_path, mode='r') as submission_file:
            reader = csv.DictReader(submission_file, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
            for row in reader:
                key = (row['identifier1'], row['identifier2'])
                submission[key] = float(row['prediction'])

        # Load truth file
        truth_file_name = language + '/labels.tsv'
        # truth_dir = os.path.join(input_dir, 'ref')
        # truth_path = os.path.join(truth_dir, truth_file_name)
        truth_path = os.path.join(input_dir2, truth_file_name)
        # if not os.path.exists(truth_path):
        #     message = "Error: Expected truth file '{0}', found files {1}"
        #     sys.exit(message.format(truth_file_name, os.listdir(truth_dir)))

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
        # print(truth_values)
        score = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
        scores[language] = score

    # Calculate the average score
    average_score = np.mean([scores[language] for language in languages])
    scores['average'] = average_score

    return average_score, scores



if __name__ == '__main__':
    # file path
    current_directory = os.path.dirname(__file__)
    output_dir = current_directory + '/result/'

    # set random number seed
    seed = 1
    seedConfig(seed)

    # the hyperparameters required for training
    learning_rate = 1e-2
    epoch_num = 50
    batch_size = 128
    num_labels = 4

    # organize the dataset
    train_dataset = TextDataset(train_input_list, train_language_list, train_label_list, train_ident1_list, train_ident2_list)
    valid_dataset = TextDataset(dev_input_list, dev_language_list, dev_label_list, dev_ident1_list, dev_ident2_list)
    test_dataset = TextDataset(test_input_list, test_language_list, test_label_list, test_ident1_list, test_ident2_list)

    # create data loader
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=trans2TheDataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=trans2TheDataset)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=trans2TheDataset)

    # instantiate model
    model = MModel(num_labels).to(device)

    # define loss function, optimizer and lr_scheduler
    # weighted cross entropy loss function
    # weights = torch.tensor([4.3, 6.7, 5.0, 1.0]).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight=weights)
    loss_fn = nn.CrossEntropyLoss()
    classifier_optimizer = list(model.classifier.named_parameters())
    trans_optimizer = list(model.trans.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in trans_optimizer if not any(nd in n for nd in no_decay)],  
         'weight_decay': 0.01},
        {'params': [p for n, p in trans_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],  
         'weight_decay': 0.01},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=epoch_num * len(train_dataloader) * 0.1,
                                 num_training_steps=epoch_num * len(train_dataloader))
    
    # record the effectiveness of each epoch during the training process
    train_loss = list()

    valid_acc = list()
    valid_k_avg = []
    valid_k_all = []

    test_acc = list()
    test_k_avg = []
    test_k_all = []

    print(f'*************** Training begins, learning rate of {learning_rate}, batch size of {batch_size}, epoch of {epoch_num}, and random seed number of {seed} ****************')
    for t in range(epoch_num):
        print(f'Epoch {t + 1}/{epoch_num}\n-----------------------', flush=True)
        # start model training
        loss = train_loop(train_dataloader, model, loss_fn, optimizer, t + 1)
        train_loss.append(loss)
        # save validation set results
        record = {'chinese': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'german': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                             'prob_2': [], 'prob_3': []},
                  'english': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'norwegian': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                             'prob_2': [], 'prob_3': []},
                  'russian': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'spanish': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'swedish': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []}
                }

        acc_dev = valid_loop(valid_dataloader, model, t + 1, record)
        valid_acc.append(acc_dev)

        # save the results of each epoch in the required evaluation format
        valid_out_dir = output_dir + f'dev/{t + 1}/'
        if not os.path.exists(valid_out_dir):
            os.makedirs(valid_out_dir)
        for language in record:
            df = pd.DataFrame({'identifier1': record[language]['ident1'], 'identifier2': record[language]['ident2'],
                               'prediction': record[language]['prediction'],
                               'prob_max': record[language]['prob_max'], 'prob_0': record[language]['prob_0'],
                               'prob_1': record[language]['prob_1'], 'prob_2': record[language]['prob_2'],
                               'prob_3': record[language]['prob_3']})
            df.to_csv(valid_out_dir + language + '.tsv', index=False, sep='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')

        # calculate krippendorff on the validation set
        k_dev_avg, k_dev_all = get_krippendorff(valid_out_dir, path_dev)
        print('***************', 'avg krippendorff on the validation set: ', k_dev_avg, '***************')
        valid_k_avg.append(k_dev_avg)
        valid_k_all.append((k_dev_all))

        # save test set results
        record = {'chinese': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'german': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                             'prob_2': [], 'prob_3': []},
                  'english': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'norwegian': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'russian': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'spanish': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []},
                  'swedish': {'ident1': [], 'ident2': [], 'prediction': [], 'prob_max': [], 'prob_0': [], 'prob_1': [],
                              'prob_2': [], 'prob_3': []}
                  }

        acc_test = valid_loop(test_dataloader, model, t + 1, record)
        test_acc.append(acc_test)

        # save the results of each epoch in the required evaluation format
        test_out_dir = output_dir + f'test/{t + 1}/'
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)
        for language in record:
            df = pd.DataFrame({'identifier1': record[language]['ident1'], 'identifier2': record[language]['ident2'],
                               'prediction': record[language]['prediction'],
                               'prob_max': record[language]['prob_max'], 'prob_0': record[language]['prob_0'],
                               'prob_1': record[language]['prob_1'], 'prob_2': record[language]['prob_2'],
                               'prob_3': record[language]['prob_3']})
            df.to_csv(test_out_dir + language + '.tsv', index=False, sep='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')

        # calculate krippendorff on the validation set
        k_test_avg, k_test_all = get_krippendorff(test_out_dir, path_test)
        print('***************', 'avg krippendorff on the test set: ', k_dev_avg, '***************')
        test_k_avg.append(k_test_avg)
        test_k_all.append(k_test_all)

    print(f'*************** Training ends, learning rate of {learning_rate}, batch size of {batch_size}, epoch of {epoch_num}, and random seed number of {seed} ****************')
    print('loss for each epoch during the training process: ', train_loss)
    print('avg krippendorff on the validation set: ', valid_k_avg)
    # select the epoch corresponding to the highest krippendorff on the validation set
    best_valid_k_avg = max(valid_k_avg)
    best_epoch_k_avg = valid_k_avg.index(best_valid_k_avg)
    best_test_k_avg = test_k_avg[best_epoch_k_avg]
    best_valid_k_all = valid_k_all[best_epoch_k_avg]
    best_test_k_all = test_k_all[best_epoch_k_avg]

    print(f'*************** Select the {best_epoch_k_avg + 1} epoch ****************')
    print(f'*************** The krippendorff on the validation set is {best_valid_k_all} ****************')
    print(f'*************** The krippendorff on the test set is {best_valid_k_all} ****************')









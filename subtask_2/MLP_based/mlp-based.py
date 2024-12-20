import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from transformers import AutoTokenizer, XLMRobertaModel

import json
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import csv
from transformers import BertTokenizer, BertModel
from transformers import get_scheduler
from tqdm import tqdm


# check if GPU is available
is_available = torch.cuda.is_available()
print(f"GPU : {is_available}")
device = torch.device("cuda" if is_available else "cpu")


path_dev = 'dev/'
path_train = 'train/'

if not os.path.exists(path_dev):       
    os.makedirs(path_dev)
with ZipFile('dev.zip', 'r') as dev:
    dev.extractall(path_dev)
if not os.path.exists(path_train):
    os.makedirs(path_train)
with ZipFile('train.zip', 'r') as train:
    train.extractall(path_train)

languages = os.listdir(path_train)

label_file_paths_train = []
uses_file_paths_train = []
instance_file_paths_dev = []
uses_file_paths_dev = []

for lang in languages:
    label_file_paths_train.append(path_train + lang + '/labels.tsv')
    uses_file_paths_train.append(path_train + lang + '/uses.tsv')
    instance_file_paths_dev.append(path_dev + lang + '/instances.tsv')
    uses_file_paths_dev.append(path_dev + lang + '/uses.tsv')


# loading train labels and uses and dev instances and uses

# dictionary containing input file paths
paths = {'train_labels_list': label_file_paths_train, 'train_uses_list': uses_file_paths_train, 'dev_uses_list': uses_file_paths_dev, 'dev_instances_list': instance_file_paths_dev}
# dictionary to store the extracted data
data_dict = {'train_labels_list': [], 'train_uses_list': [], 'dev_uses_list': [], 'dev_instances_list': []}

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


# make dictionaries to map identifiers to their contexts and target token indices from the train and dev uses data

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


# merging train labels and uses into a single dataframe

train_uses_merged= []
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
    mean_disagreement = row['mean_disagreement_cleaned']
    judgments = row['judgments']  
    language = row['language']
    data_row = {'context1': context1, 'context2': context2,'index_target_token1': index_target_token1, 'index_target_token2': index_target_token2,'identifier1': identifier1_train,'identifier2': identifier2_train,'lemma': lemma,'mean_disagreement_cleaned': mean_disagreement,'judgments': judgments, 'language':language}
    
    train_uses_merged.append(data_row)

df_train_uses_merged = pd.DataFrame(train_uses_merged)


# merging dev instances and uses into a single dataframe

dev_uses_merged = []
for row in dev_instances_list:
    identifier1_dev= row['identifier1']  
    identifier2_dev = row['identifier2']
    
    # use id2context dictionary to get the corresponding context for each identifier
    context1 = dev_id2context.get(identifier1_dev)
    context2 = dev_id2context.get(identifier2_dev)

    # use id2idx dictionary to get the corresponding target token index for each identifier
    index_target_token1 = dev_id2idx.get(identifier1_dev)
    index_target_token2 = dev_id2idx.get(identifier2_dev)
            
    lemma = row['lemma']
  
    language = row['language']
    data_row = {'context1': context1, 'context2': context2,'index_target_token1': index_target_token1, 'index_target_token2': index_target_token2,'identifier1': identifier1_dev,'identifier2': identifier2_dev,'lemma': lemma, 'language':language}
    
    dev_uses_merged.append(data_row)
    
df_dev_uses_merged = pd.DataFrame(dev_uses_merged) 


# define and load the tokenizer and model for XLM-RoBERTa. you need to change the local model address

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)

# define and load the tokenizer and model for mBERT
# tokenizer = AutoTokenizer.from_pretrained("bert-base-mult")
# model = BertModel.from_pretrained("bert-base-mult").to(device)


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
    inputs = tokenizer(context, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    
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
        print(f"Error: Target token indices not found within the specified range for context: '{context}' and index: '{index}'")
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
            print(f"Truncation failed: input sequence length ({len(input_ids[0])}) exceeds the maximum token limit for context: '{context}' and index: '{index}'")
            return None
    
    # extract the subwords for the target word
    extracted_subwords = [tokens[i] for i, value in enumerate(subwords_bool_mask) if value]
    
    if not extracted_subwords:
        print(f"Error: no subwords extracted for the target word in context: '{context}' and index: '{index}'")
        return None
        
    with torch.no_grad():
        outputs = model(inputs['input_ids'].to(device))  # get embeddings for the truncated input

    # embeddings for all tokens in the truncated input
    embeddings = outputs.last_hidden_state[0]

    # embeddings for target token
    target_embeddings = embeddings[subwords_bool_mask] 
    
    if target_embeddings.size(0) == 0:
        print(f"error: no embeddings found for the target token in context: '{context}' and index: '{index}'")
        return None
     
    # aggregated target token embedding
    target_embeddings_nump = target_embeddings.mean(dim=0).cpu().numpy()
    # target_embeddings_nump = np.concatenate(( target_embeddings_nump, embeddings[0].cpu().numpy()))

    return target_embeddings_nump


dataframes = [df_train_uses_merged, df_dev_uses_merged]
file_names = ['subtask2_train_embeddings.npz', 'subtask2_dev_embeddings.npz']

# getting target token embeddings for contexts in train and dev 
for df, file_name in zip(dataframes, file_names):
    id2embedding = {}

    for _, row in df.iterrows():
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
    
    
dataframes = [df_train_uses_merged, df_dev_uses_merged]
file_names = ['subtask2_train_embeddings.npz', 'subtask2_dev_embeddings.npz']
embeddings_lists = [[], []]

# retrieve the context embeddings using the identifiers from the dataframe
for df, file_name, embeddings in zip(dataframes, file_names, embeddings_lists):
    loaded_embeddings = np.load(file_name)
    for _, row in df.iterrows():
        try:
            context_embedding1 = loaded_embeddings[row['identifier1']]
            context_embedding2 = loaded_embeddings[row['identifier2']]
            # concatenate the embeddings to form a single feature vector
            concatenated_emb = np.concatenate((context_embedding1, context_embedding2))
            embeddings.append(concatenated_emb)
        except KeyError as e:
            print(f"KeyError: {e}. Identifier not found in embeddings file.")
            continue

# convert the lists of feature vectors to numpy arrays (feature matrices)
train_embeddings = np.array(embeddings_lists[0])
dev_embeddings = np.array(embeddings_lists[1])


df_train_uses_merged['mean_disagreement_cleaned'] = df_train_uses_merged['mean_disagreement_cleaned'].astype(float)


# define features and target variable for the model

X_train = train_embeddings
y_train = df_train_uses_merged['mean_disagreement_cleaned'].values
X_dev = dev_embeddings


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
    def __init__(self, all_embedding, all_label):
        self.all_embedding = all_embedding
        self.all_label = all_label

    def __getitem__(self, index):
        embedding = self.all_embedding[index]
        label = self.all_label[index]
        return embedding, label
    
    def __len__(self):
        return len(self.all_label)

    
# define MLP model

class MModel(nn.Module):

    def __init__(self):
        super(MModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(768*2, 1)
    
    def forward(self, embeddings):        
        
        embeddings = self.dropout(embeddings)
        logits = self.regression(embeddings)        
        return logits.squeeze(-1)

# training process

def train_loop(train_dataloader, model, loss_fn, optimizer, epoch):
    total_loss_train = 0
    size = len(train_dataloader.dataset)
    model.train()
    for inputs, labels in tqdm(train_dataloader, colour='green'):    
        labels = labels.to(device)
        logits = model(torch.Tensor(inputs).to(device))
        loss = loss_fn(logits.float(), labels.float())
        total_loss_train += loss.item()

        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()  
        lr_scheduler.step()  
                
    print(f'Epoch {epoch}    Loss: {total_loss_train / size:.5f}', flush=True)
    return total_loss_train


if __name__ == '__main__':
    # set random number seed
    seed = 1
    seedConfig(seed)

    # the hyperparameters required for training
    learning_rate = 1e-2
    epoch_num = 200
    batch_size = 32

    # instantiate model
    model = MModel().to(device)

    # organize the dataset
    train_dataset = TextDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    # define loss function, optimizer and lr_scheduler
    loss_fn = nn.MSELoss()
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in list(model.named_parameters()) if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
        {'params': [p for n, p in list(model.named_parameters()) if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=epoch_num*len(train_dataloader)*0.1, num_training_steps=epoch_num*len(train_dataloader))

    # start model training
    for t in range(epoch_num):
        print(f'Epoch {t + 1}/{epoch_num}\n-----------------------', flush=True)
        loss = train_loop(train_dataloader, model, loss_fn, optimizer, t+1)

    # predicting on the dev data per language
    for language, group in df_dev_uses_merged.groupby('language'):
        dev_indices = group.index
        X_dev = dev_embeddings[dev_indices]
        
        # predict using the fitted model
        y_pred = []
        model.eval()
        for dev_embedding in X_dev:
            y_pred.append(model(torch.tensor(dev_embedding).to(device)).cpu().detach().numpy())
        
        # add predictions to the dataframe
        df_dev_uses_merged.loc[dev_indices, 'prediction'] = y_pred
        
        
    # create answer file in required format for codalab
    out_dir = 'answer/'
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    answer_df = df_dev_uses_merged[['identifier1', 'identifier2', 'prediction', 'language']]
    answer_df = answer_df.reset_index(drop= True)
    for i in list(answer_df["language"].value_counts().index):
        df_temp = answer_df[answer_df["language"]==i]
        df_temp = df_temp.drop('language', axis=1)
        df_temp.to_csv('answer/' +i +'.tsv',index = False, sep='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')

    with ZipFile('answer.zip', 'w') as zipf:
        for root, _, files in os.walk(out_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
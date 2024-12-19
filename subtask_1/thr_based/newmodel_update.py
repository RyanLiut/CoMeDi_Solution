'''
Threshold-based main python file, adopted by COMEDI official baseline code.
Author: Juniper Zhu Liu
Time: 2024.11-12
'''

import pandas as pd
import numpy as np
import os
import csv
from zipfile import ZipFile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import krippendorff
import torch
from scipy.optimize import minimize
from transformers import AutoTokenizer, XLMRobertaModel, BertModel, AutoModelForMaskedLM
import transformers
import sys
from tqdm import tqdm
from scipy import stats
import subprocess




# In[2]:


path_dev = '/home/liuzhu/CoMeDi/subtask_1/prompt/dev/'
path_test = "/home/liuzhu/CoMeDi/subtask_1/prompt/test_hidden1/"
path_train = '/home/liuzhu/CoMeDi/subtask_1/prompt/train/'

STANDARD_type = "std" # whether to standardization
LAYER_ID = 12
model_name = "xlm-roberta-base"
[_, LAYER_ID, STANDARD_type, model_name] = sys.argv
LAYER_ID = int(LAYER_ID)
print(STANDARD_type)
out_dir = f"/home/liuzhu/CoMeDi/subtask_1/prompt/results/answer_mah_{STANDARD_type}_{model_name}_L{LAYER_ID}"

print(out_dir)
# In[3]:

if not os.path.exists(out_dir):      
    os.makedirs(out_dir)
if not os.path.exists(out_dir + "/dev/"):      
    os.makedirs(out_dir + "/dev/")
if not os.path.exists(out_dir + "/test/"):      
    os.makedirs(out_dir + "/test/")

# In[4]:


languages = os.listdir(path_train)


# In[5]:


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


# In[6]:


# loading train labels and uses and dev instances and uses

# dictionary containing input file paths
paths = {'train_labels_list': label_file_paths_train, 'train_uses_list': uses_file_paths_train, 'dev_uses_list': uses_file_paths_dev, 'dev_instances_list': instance_file_paths_dev, 'test_uses_list': uses_file_paths_test, 'test_instances_list': instance_file_paths_test}
# dictionary to store the extracted data
data_dict = {'train_labels_list': [], 'train_uses_list': [], 'dev_uses_list': [], 'dev_instances_list': [], 'test_uses_list': [], 'test_instances_list': []}

for save_path, path_list in paths.items():
    for path in path_list:
        with open(path, encoding='utf-8') as tsvfile:
            language = path.split('/')[-2]
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


# In[7]:


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
test_id2context, test_id2idx = create_mappings(test_uses_list)

# In[8]:


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
    judgments = eval(row['judgments'])
    # print(judgments) 
    median_cleaned = row['median_cleaned']
     
    language = row['language']
    data_row = {'context1': context1, 'context2': context2,'index_target_token1': index_target_token1, 'index_target_token2': index_target_token2,'identifier1': identifier1_train,'identifier2': identifier2_train,'lemma': lemma,'median_cleaned': median_cleaned,'judgments': judgments, 'language':language}
    
    train_uses_merged.append(data_row)

df_train_uses_merged = pd.DataFrame(train_uses_merged)


# In[9]:


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

test_uses_merged = []
for row in test_instances_list:
    identifier1_test= row['identifier1']  
    identifier2_test = row['identifier2']
    
    # use id2context dictionary to get the corresponding context for each identifier
    context1 = test_id2context.get(identifier1_test)
    context2 = test_id2context.get(identifier2_test)

    # use id2idx dictionary to get the corresponding target token index for each identifier
    index_target_token1 = test_id2idx.get(identifier1_test)
    index_target_token2 = test_id2idx.get(identifier2_test)
            
    lemma = row['lemma']
  
    language = row['language']
    data_row = {'context1': context1, 'context2': context2,'index_target_token1': index_target_token1, 'index_target_token2': index_target_token2,'identifier1': identifier1_test,'identifier2': identifier2_test,'lemma': lemma, 'language':language}
    
    test_uses_merged.append(data_row)
    
df_test_uses_merged = pd.DataFrame(test_uses_merged) 
# In[10]:


# define and load the tokenizer and model for XLM-RoBERTa

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(f"/data64/private/liuzhu/models/{model_name}")
if model_name in ["bert-base-multilingual-cased", "bert-large-uncased"]:
    model = BertModel.from_pretrained(f"/data64/private/liuzhu/models/{model_name}").to(device)
elif model_name in ["xlm-roberta-base", "xlm-roberta-large"]:
    model = AutoModelForMaskedLM.from_pretrained(f"/data64/private/liuzhu/models/{model_name}").to(device)
    # model = XLMRobertaModel.from_pretrained(f"/data64/private/liuzhu/models/{model_name}").to(device)
elif model_name in ["Llama-7b-hf"]:
    model = transformers.AutoModel.from_pretrained(f"/data64/private/liuzhu/models/{model_name}", device_map="auto").half()
# model = BertModel.from_pretrained(f"/data64/private/liuzhu/models/{model_name}").to(device)
# tokenizer = transformers.AutoTokenizer.from_pretrained(f"/data61/liuzhu/LLM/llama-main/{model_name}")



# In[11]:


# calculate truncation indices for a sequence of tokens to ensure that the target subwords are preserved

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
            print(f"Truncation failed: input seq len ({len(input_ids[0])}) exceeds the maximum token limit for context: '{context}' and index: '{index}'")
            return None
    
    # extract the subwords for the target word
    extracted_subwords = [tokens[i] for i, value in enumerate(subwords_bool_mask) if value]
    
    if not extracted_subwords:
        print(f"Error: no subwords extracted for the target word in context: '{context}' and index: '{index}'")
        return None
        
    with torch.no_grad():
        outputs = model(inputs['input_ids'].to("cuda"), output_hidden_states=True)  # get embeddings for the truncated input

    # embeddings for all tokens in the truncated input
    # embeddings = outputs.last_hidden_state[0]
    # print(embeddings.shape)

    embeddings = outputs.hidden_states[LAYER_ID][0]
    # print(embeddings.shape)

    # embeddings for target token
    try:
        # print(len(subwords_bool_mask))
        # print(subwords_bool_mask)
        target_embeddings = embeddings[subwords_bool_mask] 
    except:
        print("here")
    
    if target_embeddings.size(0) == 0:
        print(f"error: no embeddings found for the target token in context: '{context}' and index: '{index}'")
        return None
     
    # aggregated target token embedding
    target_embeddings_nump = target_embeddings.mean(dim=0).cpu().numpy()

    return target_embeddings_nump


# In[16]:
# out_dir = 'answer_large/'
if not os.path.exists(out_dir):
        os.mkdir(out_dir)

dataframes = [df_dev_uses_merged, df_test_uses_merged, df_train_uses_merged]
file_names = [out_dir+'/subtask1_dev_embeddings_ani.npz', out_dir+'/subtask1_test_embeddings_ani.npz', out_dir+'/subtask1_train_embeddings_ani.npz']

'''
# getting target token embeddings for contexts in train and dev 
for df, file_name in zip(dataframes, file_names):
    id2embedding = {}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        identifier1 = row['identifier1']
        identifier2 = row['identifier2']
        
        if identifier1 not in id2embedding:
            embedding1 = get_target_token_embedding(row['context1'], row['index_target_token1'])
            id2embedding[(identifier1, row['language'], "1")] = embedding1
        
        if identifier2 not in id2embedding:
            embedding2 = get_target_token_embedding(row['context2'], row['index_target_token2'])
            id2embedding[(identifier2, row['language'], "2")] = embedding2
    
    # store embeddings in a .npz file using identifiers as keys
    # [TODO 1] anisotropy removal
    keys = list(id2embedding.keys())
    embeddings = np.stack([id2embedding[k] for k in keys], axis=0)

    group_info = [(k[1], k[2]) for k in keys]
    group_set = set(group_info)

    for group in group_set:
        group_indices = [i for i, g in enumerate(group_info) if g == group]
        group_embeddings = embeddings[group_indices]
        emb_mean = np.mean(group_embeddings, axis=0)
        emb_std = np.std(group_embeddings, axis=0) + 1e-6
        embeddings[group_indices] = (group_embeddings - emb_mean) / emb_std

    id2embedding = {k[0]:e for k,e in zip(keys, embeddings)}

    # all_embeddings = np.stack([v for k,v in id2embedding.items()], axis=0)
    # emb_mean = np.mean(all_embeddings,axis=0)
    # emb_std = np.std(all_embeddings,axis=0) + 1e-6
    # print(all_embeddings.shape)
    # all_embeddings = (all_embeddings - emb_mean) / emb_std
    # id2embedding = {k: all_embeddings[i] for i, k in enumerate(id2embedding.keys())}
    np.savez(file_name, **id2embedding)
'''

# '''
# [TODO 2] parallel 
# getting target token embeddings for contexts in train and dev 
for df, file_name in zip(dataframes, file_names):
    id2embedding = {}
    languages = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        identifier1 = row['identifier1']
        identifier2 = row['identifier2']
        languages.append(row['language'])
        
        if identifier1 not in id2embedding:
            embedding1 = get_target_token_embedding(row['context1'], row['index_target_token1'])
            id2embedding[identifier1] = embedding1
        
        if identifier2 not in id2embedding:
            embedding2 = get_target_token_embedding(row['context2'], row['index_target_token2'])
            id2embedding[identifier2] = embedding2

    all_embeddings = np.stack([v for k,v in id2embedding.items()], axis=0)
    mean_embeddings = np.mean(all_embeddings, axis=0)
    std_embeddings = np.std(all_embeddings, axis=0) + 1e-6
    if STANDARD_type == "std":
        all_embeddings = (all_embeddings - mean_embeddings ) / std_embeddings
    elif STANDARD_type == "centering":
        all_embeddings = (all_embeddings - mean_embeddings )
    elif STANDARD_type == "PCArem":
        pca = PCA(n_components=1, whiten=True)
        pca.fit(all_embeddings)
        all_embeddings -= np.sum(all_embeddings * pca.components_[0], axis=1, keepdims=True) @ pca.components_[0].reshape(1,-1)
    elif STANDARD_type == "whitening":
        # Ref: https://kexue.fm/archives/8069
        def compute_kernel_bias(vecs):
            """计算kernel和bias
            vecs.shape = [num_samples, embedding_size]，
            最后的变换：y = (x + bias).dot(kernel)
            """
            mu = vecs.mean(axis=0, keepdims=True)
            cov = np.cov(vecs.T)
            u, s, vh = np.linalg.svd(cov)
            W = np.dot(u, np.diag(1 / np.sqrt(s)))
            return W, -mu
        kernel, bias = compute_kernel_bias(all_embeddings)
        all_embeddings = (all_embeddings + bias).dot(kernel)
    
    else:
        print("There is no standardization!")

    id2embedding = {k:v for k,v in zip(id2embedding.keys(), all_embeddings)}
    
    np.savez(file_name, **id2embedding)
# '''
''' 
print("Standardization within each language group!")
for df, file_name in zip(dataframes, file_names):
    id2embedding = {}
    languages = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        identifier1 = row['identifier1']
        identifier2 = row['identifier2']
        language = row['language']
        languages.append(language)
        
        if identifier1 not in id2embedding:
            embedding1 = get_target_token_embedding(row['context1'], row['index_target_token1'])
            id2embedding[identifier1] = (embedding1, language)
        
        if identifier2 not in id2embedding:
            embedding2 = get_target_token_embedding(row['context2'], row['index_target_token2'])
            id2embedding[identifier2] = (embedding2, language)

    if STANDARD:
        language_groups = {}
        language_ids = {}
        
        # 根据语言分组
        for k, (v, lang) in id2embedding.items():
            if lang not in language_groups:
                language_groups[lang] = []
                language_ids[lang] = []
            language_groups[lang].append(v)
            language_ids[lang].append(k)
        
        normalized_embeddings = {}
        
        # 对每组语言的嵌入进行归一化
        for lang, embeddings in language_groups.items():
            all_embeddings = np.stack(embeddings, axis=0)
            mean_embeddings = np.mean(all_embeddings, axis=0)
            std_embeddings = np.std(all_embeddings, axis=0) + 1e-6
            normalized_embeddings_for_lang = (all_embeddings - mean_embeddings) / std_embeddings
            
            for i, kk in enumerate(language_ids[lang]):
                normalized_embeddings[kk] = normalized_embeddings_for_lang[i]
        
        id2embedding = normalized_embeddings
    np.savez(file_name, **id2embedding)
'''



# In[23]:
cosine_similarities_lists = [[], [], []]

# iterate over the lists to compute and store cosine similarities
for df, file_name, cosine_similarities in zip(dataframes, file_names, cosine_similarities_lists):
    loaded_embeddings = np.load(file_name)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            context_embedding1 = loaded_embeddings[row['identifier1']] 
            context_embedding2 = loaded_embeddings[row['identifier2']]
            cosine_sim = cosine_similarity([context_embedding1], [context_embedding2])[0][0]
            cosine_similarities.append(cosine_sim)
        except KeyError as e:
            print(f"KeyError: {e}. row not there")
            continue
    # add the cosine similarities to the dataFrame
    df['cosine_similarity'] = cosine_similarities


# In[24]:


df_train_uses_merged['median_cleaned'] = df_train_uses_merged['median_cleaned'].astype(float)
print("Correlation between similarity and label:")
print(df_train_uses_merged['median_cleaned'].corr(df_train_uses_merged['cosine_similarity']))


# In[25]:


# input features for the threshold model

median_cleaned_train = df_train_uses_merged['median_cleaned'].tolist()
cosine_sim_train = df_train_uses_merged['cosine_similarity'].tolist()
cosine_sim_dev = df_dev_uses_merged['cosine_similarity']
cosine_sim_test = df_test_uses_merged['cosine_similarity']


# In[26]:


# threshold model 

def calc_threshold(cosine_sim_train, median_cleaned_train, n=3):
    min_sim = float(min(cosine_sim_train))
    max_sim = float(max(cosine_sim_train))
    delta = (max_sim - min_sim) / (n + 1)
    # initial bins
    bins = [min_sim + delta*(i+1) for i in range(n)]
    
    # loss function
    def min_loss(bins, cos_sim, y):
        bins = sorted([-np.inf] + list(bins) + [np.inf])
        binned_similarities = pd.cut(cos_sim, bins=bins, labels=[1.0, 2.0, 3.0, 4.0])
        y_pred = binned_similarities.tolist()
        y = [float(i) for i in y]
        data = [y, y_pred]
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
        return 1 - alpha
    
    # optimizing bin edges
    result = minimize(min_loss, bins, args=(cosine_sim_train, median_cleaned_train), method='nelder-mead')
    optimized_bins = sorted([-np.inf] + result.x.tolist() + [np.inf])
    
    return optimized_bins


# In[27]:


# calculate optimised bins(thresholds) per language

grouped_lang_df = df_train_uses_merged.groupby('language')

optimized_bins_dict = {}

for language, group in grouped_lang_df:
    median_cleaned_train = group['median_cleaned'].values.tolist()
    cosine_sim_train = group['cosine_similarity'].values.tolist()
    
    optimized_bins = calc_threshold(cosine_sim_train, median_cleaned_train)
    optimized_bins_dict[language] = optimized_bins


# In[28]:


# get predictions on dev based on the optimised bins

predictions = []
predictions_sim = []
for _, row in df_dev_uses_merged.iterrows():
    language = row['language']
    cosine_sim_dev = row['cosine_similarity']
    optimized_bins = optimized_bins_dict[language]
    prediction = pd.cut([cosine_sim_dev], bins=optimized_bins, labels=[1.0, 2.0, 3.0, 4.0])
    predictions.append(prediction[0])
    predictions_sim.append(cosine_sim_dev)

df_dev_uses_merged['prediction'] = predictions
df_dev_uses_merged['prediction_sim'] = predictions_sim

predictions = []
predictions_sim = []
for _, row in df_test_uses_merged.iterrows():
    language = row['language']
    cosine_sim_test = row['cosine_similarity']
    optimized_bins = optimized_bins_dict[language]
    prediction = pd.cut([cosine_sim_test], bins=optimized_bins, labels=[1.0, 2.0, 3.0, 4.0])
    predictions.append(prediction[0])
    predictions_sim.append(cosine_sim_test)

df_test_uses_merged['prediction'] = predictions
df_test_uses_merged['prediction_sim'] = predictions_sim
# In[29]:


# create output in required format for codalab
answer_df = df_dev_uses_merged[['identifier1', 'identifier2', 'prediction', 'prediction_sim', 'language']]
answer_df = answer_df.reset_index(drop= True)
for i in list(answer_df["language"].value_counts().index):
    df_temp = answer_df[answer_df["language"]==i]
    df_temp = df_temp.drop('language', axis=1)
    df_temp.to_csv(out_dir +'/dev/' +i +'.tsv',index = False, sep='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')
# print(out_dir)
# with ZipFile('answer_ani.zip', 'w') as zipf:
#     for root, _, files in os.walk(out_dir):
#         for file in files:
#             zipf.write(os.path.join(root, file), arcname=file)

answer_df = df_test_uses_merged[['identifier1', 'identifier2', 'prediction', 'prediction_sim', 'language']]
answer_df = answer_df.reset_index(drop= True)
for i in list(answer_df["language"].value_counts().index):
    df_temp = answer_df[answer_df["language"]==i]
    df_temp = df_temp.drop('language', axis=1)
    df_temp.to_csv(out_dir +'/test/' +i +'.tsv',index = False, sep='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')
print(out_dir)

os.system(f'python /home/liuzhu/CoMeDi/subtask_1/evaluation.py {out_dir}/test')
os.system(f'python /home/liuzhu/CoMeDi/subtask_1/evaluation.py {out_dir}/dev')
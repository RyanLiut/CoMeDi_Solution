# CoMeDi_Solution ![Static Badge](https://img.shields.io/badge/license-MIT-green) ![Static Badge](https://img.shields.io/badge/COLING-2025-blue)

Our Solution to [CoMeDi](https://comedinlp.github.io/): Context and Meaning—Navigating Disagreements in NLP Annotations Workshop to be held in conjunction with [COLING 2025](https://coling2025.org/program/) in Abu Dhabi.

## Subtask 1
### 🌟 MLP-based method for Subtask 1
A classifier composed of two linear layers with ReLU activation function, is trained using a cross-entropy loss function.  
Use the following command to start mlp-based method training:  
```  
python mlp-based.py
```
⚠ Before training, please specify the training data path, model path, and configure specific training parameters:  
```python 
# raw data path
path_dev = '../data/dev/'
path_train = '../data/train/'
path_test = '../data/test/'

# your local model path
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)

# the hyperparameters required for training
learning_rate = 1e-2
epoch_num = 50
batch_size = 128
```
### 🌟 Threshold-based method for Subtask 1
#### Non prompt+LLM method:
Simply run the file 'threshold.ipynb' directly.
#### prompt+LLM method:
You need to change the file path, as follows:
```python 
path_dev = '../prompt_data/dev/'
path_train = '../prompt_data/train/'
```
if prompt data is not generated in advance, please run:
```
python subtask_1/prompt_data/add_prompt.py
```
⚠ Specify the model path, model name and selected model layers:   
```python
# define and load the tokenizer and model for different choices
# models can be downloaded from huggingface
MODEL_PATH = "" # offline model path
model_name = "xlm-roberta-base"
LAYER_ID = -1 # choose which layer to extract representations
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + f"/{model_name}")
if model_name in ["bert-base-multilingual-cased", "bert-large-uncased"]:
    model = BertModel.from_pretrained(MODEL_PATH + f"/{model_name}").to(device)
elif model_name in ["xlm-roberta-base", "xlm-roberta-large"]:
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH + f"/{model_name}").to(device)
elif model_name in ["Llama-7b-hf"]:
    model = transformers.AutoModel.from_pretrained(MODEL_PATH + f"/{model_name}", device_map="auto").half()
```
⚠ At the same time, please select the standardization method:  
```python
# Anisotropy Removal
dataframes = [df_train_uses_merged, df_dev_uses_merged]
file_names = ['subtask1_train_embeddings.npz', 'subtask1_dev_embeddings.npz']
STANDARD_type = "std" # techniques for anisotropy removal.
```
## Subtask 2

### 🌟 MLP-based method for Subtask 2
Train a linear layer using the MSE loss function for regression prediction.  
Use the following command to start mlp-based method training:  
```  
python mlp-based.py
```
⚠ Before training, please specify the training data path, model path, and configure specific training parameters:  
```python 
# raw data path
path_dev = '../data/dev/'
path_train = '../data/train/'

# your local model path
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)

# the hyperparameters required for training
learning_rate = 1e-2
epoch_num = 200
batch_size = 32
```
### 🌟 Threshold-based method for Subtask 2
The ground truth of training data needs to be calculated manually, that is, the mode of annotator is used.  
```python 
judgments = eval(row['judgments'])
# print(judgments) 
median_cleaned = max(stats.mode(judgments).mode, 1)
```  
Then merge the different model result files and run the code:  
``` 
python subtask_2/thr_based/ensemble.py
```
⚠ Relevant file parameters and fused metric need to be changed:
```python 
ROOT_DIR = "CoMeDi_Solution"
out_dir = ROOT_DIR + "/subtask_2/answer_ensemble/"
# Multiple results to ensemble
submission_path_list = [
    "/ANSWER_DIR_0",
    "/ANSWER_DIR_1",
    "/ANSWER_DIR_2",
]
metric = "STD" # Other choices: "MDP", "VR"
```  
## Others
🌟 some results: [GoogleDrive](https://drive.google.com/drive/folders/1EQ6SZftkrdIEjY8nXvzG7WSqwHRal8LH?usp=sharing).

## Citation
```
@misc{liu2024juniperliu,
    title={JuniperLiu at CoMeDi Shared Task: Models as Annotators in Lexical Semantics Disagreements},
    author={Zhu Liu and Zhen Hu and Ying Liu},
    year={2024},
    eprint={2411.12147},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgement
The code is based on the official code released by [CoMeDi](https://comedinlp.github.io/). We would like to express our gratitude to the authors for the work and code.

If you have any problem or suggestion, please feel free to pose an issue or concat us at [liuzhu22@mails.tsinghua.edu.cn](mailto:liuzhu22@mails.tsinghua.edu.cn).


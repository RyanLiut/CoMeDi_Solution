# CoMeDi_Solution ![Static Badge](https://img.shields.io/badge/license-MIT-green) ![Static Badge](https://img.shields.io/badge/COLING-2025-blue)

Our Solution to [CoMeDi](https://comedinlp.github.io/). The code is soon.

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
⚡ if prompt data is not generated in advance, please run:
```
python subtask_1/prompt_data/add_prompt.py
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
ROOT_DIR = "/home/liuzhu/CoMeDi_Solution"
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
🌟 our paper link:  
🌟 some results:
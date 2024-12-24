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

## Others
🌟 our paper link:  
🌟 some results:
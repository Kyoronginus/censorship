import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import re
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import TrainerCallback

# Function to read all CSV files from a given folder
def read_csv_folder(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=';', on_bad_lines='skip')
            dataframes.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return pd.concat(dataframes, ignore_index=True)

# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuationF
        return text
    else:
        return ''
    
# Convert labels into lists of integers for multi-label classification
def process_labels(label_str):
    if isinstance(label_str, str):
        labels = list(map(int, label_str.split(',')))
        return labels
    else:
        return []
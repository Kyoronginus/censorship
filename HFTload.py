import torch
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer
import glob
import os
from utils.evaluation_utils import plot_heatmap, print_classification_report

# Function to read all CSV files from a given folder
def read_csv_folder(folder_path):
    # Use glob to get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # List to store DataFrames
    dataframes = []
    
    # Iterate through the CSV files and read them
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=';', on_bad_lines='skip')
            dataframes.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Concatenate all DataFrames
    return pd.concat(dataframes, ignore_index=True)

# Define the folder paths
indo_folder = r'C:\Users\tohru\Documents\programming\censorship\indo'
eng_folder = r'C:\Users\tohru\Documents\programming\censorship\eng'


# Read the CSV files from both folders
indo_data = read_csv_folder(indo_folder)
eng_data = read_csv_folder(eng_folder)

data = pd.concat([indo_data, eng_data], ignore_index=True)

# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    return ''

# Apply preprocessing
data['text'] = data['text'].apply(preprocess)

# Convert labels
def process_labels(label_str):
    if isinstance(label_str, str):
        labels = list(map(int, label_str.split(',')))
        return labels
    return []

data['label'] = data['label'].apply(process_labels)

# MultiLabelBinarizer for one-hot encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['label'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.2, random_state=69)

# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

print("Model and tokenizer loaded from './saved_model'.")

# Tokenize input
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

# Create CustomDataset for test data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = CustomDataset(test_encodings, y_test)

# Predict on the test set
def predict_on_test_set(test_dataset):
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
        with torch.no_grad():
            outputs = model(**{key: val.to(model.device) for key, val in batch.items()})
            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            predictions.extend(preds)
    
    return np.array(predictions)

# Get predictions and calculate accuracy
predictions = predict_on_test_set(test_dataset)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)
accuracy = accuracy_score(y_test, binary_predictions)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Tampilkan Heatmap
plot_heatmap(y_test, binary_predictions, mlb.classes_)

# Tampilkan Classification Report
print_classification_report(y_test, binary_predictions, mlb.classes_)

# Predict function for a new sentence
def predict_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).detach().numpy()
    
    label_texts = []
    for i, value in enumerate(predictions[0]):
        percentage = value * 100
        if mlb.classes_[i] == 0:
            label_texts.append(f'Suitable: {percentage:.2f}%')
        elif mlb.classes_[i] == 1:
            label_texts.append(f'Kasar: {percentage:.2f}%')
        elif mlb.classes_[i] == 2:
            label_texts.append(f'Cabul: {percentage:.2f}%')

    return ', '.join(label_texts)

def chunk_text(text, chunk_size=3, overlap=1):
    """
    Split the text into chunks of `chunk_size` with overlap of `overlap` words.
    This creates a sliding window effect to capture context.
    """
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def mask_kasar_cabul_chunks(sentence, chunk_size=3, overlap=1):
    """
    Process a sentence by breaking it into chunks and masking chunks classified as 'kasar' or 'cabul'.
    """
    original_sentence = sentence
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())  # Remove punctuation and lowercase
    chunks = chunk_text(sentence, chunk_size, overlap)  # Split sentence into chunks

    # Iterate through each chunk and predict its class
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits).detach().numpy()
        
        # Get probabilities for 'kasar' and 'cabul'
        kasar_prob = predictions[0][mlb.classes_.tolist().index(1)]  # Index 1 corresponds to 'kasar'
        cabul_prob = predictions[0][mlb.classes_.tolist().index(2)]  # Index 2 corresponds to 'cabul'

        # Use threshold to classify chunks as 'kasar' or 'cabul'
        threshold = 0.3
        if kasar_prob > threshold:
            original_sentence = original_sentence.replace(chunk, '*****')
        elif cabul_prob > threshold:
            original_sentence = original_sentence.replace(chunk, '#####')

    return original_sentence

# Updated predict_and_censor function using chunk-based censorship
def predict_and_censor(sentence):
    try:
        censored_sentence = mask_kasar_cabul_chunks(sentence, chunk_size=3, overlap=1)
        return censored_sentence
    except Exception as e:
        print(f"Error during prediction and censoring: {e}")
        return "An error occurred. Please try again."

# Interactive input for continuous prediction with censoring
while True:
    try:
        sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
        if sentence.lower() == 'exit':
            break
        print(predict_and_censor(sentence))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function for predicting a sentence and censoring the kasar parts
def predict_and_censor(sentence):
    try:
        censored_sentence = mask_kasar_words(sentence)
        return censored_sentence
    except Exception as e:
        print(f"Error during prediction and censoring: {e}")
        return "An error occurred. Please try again."

# Interactive input for continuous prediction with censoring
while True:
    try:
        sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
        if sentence.lower() == 'exit':
            break
        print(predict_and_censor(sentence))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# # Interactive input for continuous prediction
# while True:
#     sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
#     if sentence.lower() == 'exit':
#         break
#     print(predict_sentence(sentence))

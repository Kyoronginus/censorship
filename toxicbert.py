import torch
import json
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer
import os
from utils.evaluation_utils import compute_per_label_accuracy, plot_confusion_matrix  # Impor fungsi evaluasi

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

# Define the folder paths
eng_folder = r'eng'
eng_data = read_csv_folder(eng_folder)
data = eng_data

# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    return ''

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
model = BertForSequenceClassification.from_pretrained('./saved_model_roberta')
tokenizer = BertTokenizer.from_pretrained('./saved_model_roberta')
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

# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# # Compute per-label accuracy
# accuracies = compute_per_label_accuracy(y_test, binary_predictions, mlb)
# print("Accuracy per label:")
# for label, accuracy in accuracies.items():
#     print(f"{label}: {accuracy * 100:.2f}%")

# Plot confusion matrix heatmap
plot_confusion_matrix(y_test, binary_predictions, mlb.classes_)

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
            label_texts.append(f'Inappropriate: {percentage:.2f}%')
        elif mlb.classes_[i] == 2:
            label_texts.append(f'Sexual-toned: {percentage:.2f}%')

    return ', '.join(label_texts)

# Interactive input for continuous prediction
while True:
    sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
    if sentence.lower() == 'exit':
        break
    print(predict_sentence(sentence))




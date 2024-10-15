import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer
import glob
import os
import torch

# Cek apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



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

# Load the pre-trained RoBERTa model and tokenizer
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Pindahkan model ke GPU
model.to(device)

print("Model and tokenizer loaded from 'roberta-base'.")

# Tokenize input
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

# Create CustomDataset for training and test data
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

train_dataset = CustomDataset(train_encodings, y_train)
test_dataset = CustomDataset(test_encodings, y_test)

# Training loop
def train(model, train_dataset, epochs=3, batch_size=8):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            # Pindahkan batch ke GPU
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, batch['labels'])
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} completed.")

# Train the model
train(model, train_dataset)

# Predict on the test set
def predict_on_test_set(test_dataset):
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
        # Pindahkan batch ke GPU
        batch = {key: val.to(device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            predictions.extend(preds)
    
    return np.array(predictions)

# Get predictions and calculate accuracy
predictions = predict_on_test_set(test_dataset)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)
accuracy = accuracy_score(y_test, binary_predictions)

print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Predict function for a new sentence
def predict_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Pindahkan inputs ke GPU
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    
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

# Interactive input for continuous prediction
while True:
    sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
    if sentence.lower() == 'exit':
        break
    print(predict_sentence(sentence))

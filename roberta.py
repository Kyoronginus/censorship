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
from transformers import RobertaTokenizer, RobertaForSequenceClassification

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

# Read the CSV files from both folders
eng_data = read_csv_folder(eng_folder)

# Use only English data
data = eng_data

# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuationF
        return text
    else:
        return ''

# Apply preprocessing to text column
data['text'] = data['text'].apply(preprocess)

# Convert labels into lists of integers for multi-label classification
def process_labels(label_str):
    if isinstance(label_str, str):
        labels = list(map(int, label_str.split(',')))
        return labels
    else:
        return []

data['label'] = data['label'].apply(process_labels)

# Using MultiLabelBinarizer for one-hot encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['label'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.2, random_state=69)

# Initialize the Tokenizer and Model (Inisialisasi sebelum tokenisasi)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load RoBERTa model for multi-label classification
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(mlb.classes_),  # Adjust to the number of labels in your dataset
    problem_type="multi_label_classification"  # Explicitly set multi-label classification
)

# Tokenize the input
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

# Convert to PyTorch Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # Ensure float for multi-label
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, y_train)
test_dataset = CustomDataset(test_encodings, y_test)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Optional: evaluate every epoch
    no_cuda=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the trained model and tokenizer
model.save_pretrained('./saved_model_roberta_base')
tokenizer.save_pretrained('./saved_model_roberta_base')
print("Model and tokenizer saved to './saved_model_roberta_base'.")

# Predict function for a new sentence
def predict_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    
    # Display all labels with probabilities
    label_texts = []
    for i, value in enumerate(predictions[0]):
        percentage = value * 100  # Convert to percentage
        if mlb.classes_[i] == 0:
            label_texts.append(f'Suitable: {percentage:.2f}%')
        elif mlb.classes_[i] == 1:
            label_texts.append(f'Inappropriate: {percentage:.2f}%')
        elif mlb.classes_[i] == 2:
            label_texts.append(f'Sexual-toned: {percentage:.2f}%')
    return ', '.join(label_texts)

# Predict on the test set
def predict_on_test_set(test_dataset):
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
        with torch.no_grad():
            outputs = model(**{key: val.to(model.device) for key, val in batch.items()})
            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()  # Convert to numpy array
            predictions.extend(preds)
    
    return np.array(predictions)

# Get predictions for the test set
predictions = predict_on_test_set(test_dataset)

# Apply threshold to get binary predictions
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Interactive input for continuous prediction
while True:
    sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
    if sentence.lower() == 'exit':
        break
    print(predict_sentence(sentence))

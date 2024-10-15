import torch
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset (for testing or prediction)
try:
    data1 = pd.read_csv('goblok.csv', sep=';', on_bad_lines='skip')
    data2 = pd.read_csv('kontol.csv', sep=';', on_bad_lines='skip')
    data3 = pd.read_csv('anjing.csv', sep=';', on_bad_lines='skip')
    data4 = pd.read_csv('monyet.csv', sep=';', on_bad_lines='skip')
    data6 = pd.read_csv('dataset_50_cabul_revisi.csv', sep=';', on_bad_lines='skip')

    data = pd.concat([data1, data2, data3, data4, data6], ignore_index=True)
except FileNotFoundError:
    raise Exception("The dataset file was not found.")

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

def mask_kasar_cabul_words(sentence):
    # Tokenize the input sentence and preprocess
    original_sentence = sentence
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())  # Remove punctuation and lowercase the sentence
    words = sentence.split()  # Split sentence into words
    
    # Tokenize the original sentence
    inputs = tokenizer(original_sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    
    # Get prediction probabilities
    predictions = torch.sigmoid(outputs.logits).detach().numpy()
    
    # Use thresholds for 'kasar' and 'cabul'
    threshold = 0.3
    kasar_prob = predictions[0][mlb.classes_.tolist().index(1)]  # Index 1 corresponds to 'kasar'
    cabul_prob = predictions[0][mlb.classes_.tolist().index(2)]  # Index 2 corresponds to 'cabul'
    
    # If the sentence is predicted to have 'kasar' or 'cabul' content, process the words
    if kasar_prob > threshold or cabul_prob > threshold:
        # For each word in the original sentence, check if the model classifies it as 'kasar' or 'cabul'
        for word in words:
            # Tokenize and predict each word individually
            sub_inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True, max_length=128)
            sub_outputs = model(**sub_inputs)
            sub_predictions = torch.sigmoid(sub_outputs.logits).detach().numpy()
            
            # Get the probability of 'kasar' and 'cabul' labels for each word
            sub_kasar_prob = sub_predictions[0][mlb.classes_.tolist().index(1)]  # Get 'kasar' probability
            sub_cabul_prob = sub_predictions[0][mlb.classes_.tolist().index(2)]  # Get 'cabul' probability
            
            # Replace words classified as 'kasar' or 'cabul'
            if sub_kasar_prob > threshold:
                original_sentence = original_sentence.replace(word, '*****')
            elif sub_cabul_prob > threshold:
                original_sentence = original_sentence.replace(word, '#####')
    
    return original_sentence

# Function for predicting a sentence and censoring the kasar and cabul parts
def predict_and_censor(sentence):
    try:
        censored_sentence = mask_kasar_cabul_words(sentence)
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

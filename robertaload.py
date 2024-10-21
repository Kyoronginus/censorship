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
            label_texts.append(f'Inappropriate: {percentage:.2f}%')
        elif mlb.classes_[i] == 2:
            label_texts.append(f'Sexual-toned: {percentage:.2f}%')

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
        threshold = 0.5
        if kasar_prob > threshold:
            original_sentence = original_sentence.replace(chunk, '*****')
        elif cabul_prob > threshold:
            original_sentence = original_sentence.replace(chunk, '#####')

    return original_sentence

def omit_sensor_symbols(sentence):
    """
    Remove sensor symbols ('*****' and '#####') from the sentence.
    """
    sentence = sentence.replace('*****', '').replace('#####', '')
    return sentence

def restore_sensor_symbols(sentence, first_pass_sentence):
    """
    Restore sensor symbols ('*****' and '#####') from the first pass into the final censored sentence.
    Ensure the number of chunks remains consistent between both passes.
    """
    restored_sentence = ''
    
    # Split sentences into words instead of chunks to maintain consistency
    first_pass_words = first_pass_sentence.split()
    second_pass_words = sentence.split()

    # Iterate over both word lists and restore symbols where necessary
    for i in range(len(first_pass_words)):
        if i < len(second_pass_words):  # Ensure index does not go out of range
            if '*****' in first_pass_words[i] or '#####' in first_pass_words[i]:
                restored_sentence += first_pass_words[i] + ' '
            else:
                restored_sentence += second_pass_words[i] + ' '
        else:
            # If second_pass_words is shorter, just use the first_pass_words content
            restored_sentence += first_pass_words[i] + ' '

    return restored_sentence.strip()

def predict_and_censor(sentence):
    """
    Perform censoring twice to ensure no 'kasar' or 'cabul' words are missed.
    """
    try:
        # First round of censoring
        first_censor_pass = mask_kasar_cabul_chunks(sentence, chunk_size=3, overlap=1)
        
        print("First censor : "  + (first_censor_pass))
        # Omit symbols (***** and #####) from the first censored sentence before the second pass
        sentence_without_symbols = omit_sensor_symbols(first_censor_pass)
        
        # Second round of censoring based on the result of the first pass (without symbols)
        second_censor_pass = mask_kasar_cabul_chunks(sentence_without_symbols, chunk_size=2,overlap = 1)
        
        # Restore the censor symbols from the first pass back to the final sentence
        fully_censored_sentence = restore_sensor_symbols(second_censor_pass, first_censor_pass)
        
        return second_censor_pass
    except Exception as e:
        print(f"Error during prediction and censoring: {e}")
        return "An error occurred. Please try again."

# Interactive input for continuous prediction with double censoring
while True:
    try:
        sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
        if sentence.lower() == 'exit':
            break
        print("Second censor : "  + predict_and_censor(sentence))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# # Interactive input for continuous prediction
# while True:
#     sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
#     if sentence.lower() == 'exit':
#         break
#     print(predict_sentence(sentence))

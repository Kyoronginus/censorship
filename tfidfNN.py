import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import DataLoader, Dataset

# Load the dataset
try:
    data1 = pd.read_csv('goblok.csv', sep=';', on_bad_lines='skip')
    data2 = pd.read_csv('kontol.csv', sep=';', on_bad_lines='skip')
    data3 = pd.read_csv('anjing.csv', sep=';', on_bad_lines='skip')
    data4 = pd.read_csv('monyet.csv', sep=';', on_bad_lines='skip')
    data5 = pd.read_csv('babi.csv', sep=';', on_bad_lines='skip')
    data6 = pd.read_csv('dataset_50_cabul_revisi.csv', sep=';', on_bad_lines='skip')
    data7 = pd.read_csv('english_swears.csv', sep=';', on_bad_lines='skip')
    data8 = pd.read_csv('feetBokepLabel.csv', sep=';', on_bad_lines='skip')
    data9 = pd.read_csv('feetXlabel.csv', sep=';', on_bad_lines='skip')

    data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9], ignore_index=True)
except FileNotFoundError:
    raise Exception("The dataset file was not found.")

# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()  # Ubah ke huruf kecil
        text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
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

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the text data with TF-IDF
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Custom Dataset for PyTorch
class CustomTextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create dataset and dataloaders
train_dataset = CustomTextDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomTextDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple feedforward neural network
class TextClassificationNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 128)        # Second hidden layer
        self.fc3 = nn.Linear(128, output_dim) # Output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Define model parameters
input_dim = X_train_tfidf.shape[1]  # Number of features from TF-IDF
output_dim = len(mlb.classes_)      # Number of classes

# Instantiate the model, define the loss function and the optimizer
model = TextClassificationNN(input_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()  # Since we are doing multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = torch.sigmoid(outputs).cpu().numpy()  # Sigmoid to get probabilities
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# Get predictions on test set
predictions, true_labels = evaluate_model(model, test_loader)

# Apply threshold to convert probabilities to binary labels
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(true_labels, binary_predictions)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Predict function for a new sentence using TF-IDF vectorization
def predict_sentence_nn(sentence):
    sentence = preprocess(sentence)
    sentence_tfidf = tfidf_vectorizer.transform([sentence]).toarray()
    sentence_tensor = torch.tensor(sentence_tfidf, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(sentence_tensor)
        prediction = torch.sigmoid(output).cpu().numpy()[0]  # Get probabilities

    label_texts = []
    for i, value in enumerate(prediction):
        percentage = value * 100  # Convert to percentage
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
    print(predict_sentence_nn(sentence))

# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import re
import os
import glob

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
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    return ''

# Apply preprocessing
data['text'] = data['text'].apply(preprocess)

# Convert labels from comma-separated values to lists of integers
def process_labels(label_str):
    if isinstance(label_str, str):
        labels = list(map(int, label_str.split(',')))
        return labels[0]  # Since we're using single labels (0: Suitable, 1: Kasar, 2: Cabul)
    return 0

data['label'] = data['label'].apply(process_labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=69)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # You can adjust 'ngram_range' and 'max_features'

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf.transform(X_test)

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Suitable", "Kasar", "Cabul"]))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Predict on training data
y_train_pred = model.predict(X_train_tfidf)

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='weighted')
test_recall = recall_score(y_test, y_pred, average='weighted')
test_f1 = f1_score(y_test, y_pred, average='weighted')

# Store metrics in a dictionary
metrics = {
    'Accuracy': [train_accuracy, test_accuracy],
    'Precision': [train_precision, test_precision],
    'Recall': [train_recall, test_recall],
    'F1-Score': [train_f1, test_f1]
}

# Plot the metrics
x = np.arange(len(metrics))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot for training and test metrics
train_bars = ax.bar(x - width / 2, [metrics[m][0] for m in metrics], width, label='Train', color='skyblue')
test_bars = ax.bar(x + width / 2, [metrics[m][1] for m in metrics], width, label='Test', color='orange')

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics: Train vs Test')
ax.set_xticks(x)
ax.set_xticklabels(metrics.keys())
ax.legend()

# Annotate bar heights
for bars in [train_bars, test_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3),  # Offset text
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix by row (actual class)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Display the normalized confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=["Suitable", "Kasar", "Cabul"], yticklabels=["Suitable", "Kasar", "Cabul"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Normalized)')
plt.show()




















# # Function to predict on new sentences
# def predict_sentence(sentence):
#     # Preprocess and transform the sentence into TF-IDF features
#     sentence = preprocess(sentence)
#     sentence_tfidf = tfidf.transform([sentence])
    
#     # Predict the label (0: Suitable, 1: Kasar, 2: Cabul)
#     label = model.predict(sentence_tfidf)[0]
    
#     # Optionally, if you want probabilities for each class
#     probabilities = model.predict_proba(sentence_tfidf)[0]
    
#     # Map the label to its corresponding class
#     label_mapping = {0: 'Suitable', 1: 'Kasar', 2: 'Cabul'}
    
#     # Display the percentage probabilities
#     label_texts = [
#         f"Suitable: {probabilities[0] * 100:.2f}%",
#         f"Kasar: {probabilities[1] * 100:.2f}%",
#         f"Cabul: {probabilities[2] * 100:.2f}%"
#     ]
    
#     print(", ".join(label_texts))  # Show percentage breakdown of each class
    
#     return label_mapping[label]


# # Test the function with a new sentence
# new_sentence = "You are an idiot."
# result = predict_sentence(new_sentence)
# print(f"Prediction for '{new_sentence}': {result}")

# # Continuous input for testing
# while True:
#     sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
#     if sentence.lower() == 'exit':
#         break
#     print(predict_sentence(sentence))

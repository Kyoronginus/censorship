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

# Function to predict on new sentences
def predict_sentence(sentence):
    # Preprocess and transform the sentence into TF-IDF features
    sentence = preprocess(sentence)
    sentence_tfidf = tfidf.transform([sentence])
    
    # Predict the label (0: Suitable, 1: Kasar, 2: Cabul)
    label = model.predict(sentence_tfidf)[0]
    
    # Optionally, if you want probabilities for each class
    probabilities = model.predict_proba(sentence_tfidf)[0]
    
    # Map the label to its corresponding class
    label_mapping = {0: 'Suitable', 1: 'Kasar', 2: 'Cabul'}
    
    # Display the percentage probabilities
    label_texts = [
        f"Suitable: {probabilities[0] * 100:.2f}%",
        f"Kasar: {probabilities[1] * 100:.2f}%",
        f"Cabul: {probabilities[2] * 100:.2f}%"
    ]
    
    print(", ".join(label_texts))  # Show percentage breakdown of each class
    
    return label_mapping[label]


# Test the function with a new sentence
new_sentence = "You are an idiot."
result = predict_sentence(new_sentence)
print(f"Prediction for '{new_sentence}': {result}")

# Continuous input for testing
while True:
    sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
    if sentence.lower() == 'exit':
        break
    print(predict_sentence(sentence))

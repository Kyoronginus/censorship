import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import Sequential
from tensorflow import Dense

# Load the dataset
try:
    data = pd.read_csv('dataset_anjing.csv')
except FileNotFoundError:
    raise Exception("The dataset file 'dataset_anjing.csv' was not found.")

# Simple text preprocessing
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    return text

# Apply preprocessing to text column
if 'text' in data.columns and 'label' in data.columns:
    data['text'] = data['text'].apply(preprocess)
else:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

# Encode labels (0 for non-offensive, 1 for offensive)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=69)

# Convert text to numerical features using CountVectorizer (Bag-of-Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Build a simple neural network model
model = Sequential()
model.add(Dense(16, input_dim=X_train_vec.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (offensive or non-offensive)

# Compile the model with additional metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])

# Train the model
# Use sparse matrix input for memory efficiency
model.fit(X_train_vec.toarray(), y_train, epochs=10, batch_size=10, validation_data=(X_test_vec.toarray(), y_test))

# Evaluate the model
loss, accuracy, precision, recall = model.evaluate(X_test_vec.toarray(), y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision:.2f}, Recall: {recall:.2f}')

# Predict function for a new sentence
def predict_sentence(sentence):
    sentence = preprocess(sentence)
    sentence_vec = vectorizer.transform([sentence]).toarray()  # Vectorize the sentence
    prediction = model.predict(sentence_vec)
    return 'Offensive' if prediction > 0.5 else 'Non-offensive'

# Test with new sentences
print(predict_sentence("Anjing itu lucu sekali"))  # Non-offensive
print(predict_sentence("Dasar kamu anjing"))       # Offensive

# Interactive input for continuous prediction
while True:
    sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
    if sentence.lower() == 'exit':
        break
    print(predict_sentence(sentence))

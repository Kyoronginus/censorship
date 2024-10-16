import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load the dataset with offensive labels and categories (sexual, racial, physical/mental, other)
try:
    data = pd.read_csv('dataset_offensive_with_categories.csv')  # Dataset must have 'offensive' and 'category' columns
except FileNotFoundError:
    raise Exception("The dataset file 'dataset_offensive_with_categories.csv' was not found.")

# Simple text preprocessing function
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    return text

# Apply preprocessing to text column
if 'text' in data.columns and 'offensive' in data.columns and 'category' in data.columns:
    data['text'] = data['text'].apply(preprocess)
else:
    raise ValueError("Dataset must contain 'text', 'offensive', and 'category' columns.")

# Encode offensive labels (0: Non-offensive, 1: Offensive)
label_encoder_offensive = LabelEncoder()
data['offensive'] = label_encoder_offensive.fit_transform(data['offensive'])

# Encode categories for offensive sentences (0: Sexual, 1: Racial, 2: Physical/Mental, 3: Other)
label_encoder_category = LabelEncoder()
data['category'] = label_encoder_category.fit_transform(data['category'])

# Step 1: Split data for detecting offensive/non-offensive sentences
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['offensive'], test_size=0.2, random_state=69)

# Convert text to numerical features using CountVectorizer (Bag-of-Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 2: Build a neural network model for detecting offensive/non-offensive
model_offensive = Sequential()
model_offensive.add(Dense(16, input_dim=X_train_vec.shape[1], activation='relu'))
model_offensive.add(Dense(8, activation='relu'))
model_offensive.add(Dense(1, activation='sigmoid'))  # Binary classification: offensive or non-offensive

# Compile the offensive detection model
model_offensive.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the offensive detection model
model_offensive.fit(X_train_vec.toarray(), y_train, epochs=10, batch_size=10, validation_data=(X_test_vec.toarray(), y_test))

# Step 3: Prepare data for offensive sentence categorization
# Filter offensive sentences only
offensive_data = data[data['offensive'] == 1]
X_train_off, X_test_off, y_train_off, y_test_off = train_test_split(offensive_data['text'], offensive_data['category'], test_size=0.2, random_state=69)

# Convert text to numerical features for offensive categorization
X_train_off_vec = vectorizer.fit_transform(X_train_off)
X_test_off_vec = vectorizer.transform(X_test_off)

# Step 4: Build a neural network model for offensive sentence categorization
model_category = Sequential()
model_category.add(Dense(16, input_dim=X_train_off_vec.shape[1], activation='relu'))
model_category.add(Dense(8, activation='relu'))
model_category.add(Dense(4, activation='softmax'))  # 4 categories: Sexual, Racial, Physical/Mental, Other

# Compile the category classification model
model_category.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the offensive categorization model
model_category.fit(X_train_off_vec.toarray(), y_train_off, epochs=10, batch_size=10, validation_data=(X_test_off_vec.toarray(), y_test_off))

# Predict function: first detect if the sentence is offensive, then categorize if offensive
def predict_sentence(sentence):
    sentence = preprocess(sentence)
    sentence_vec = vectorizer.transform([sentence]).toarray()  # Vectorize the sentence

    # Step 1: Check if the sentence is offensive
    offensive_prediction = model_offensive.predict(sentence_vec)[0][0]
    if offensive_prediction > 0.5:
        # Step 2: If offensive, categorize it into one of the offensive categories
        category_prediction = model_category.predict(sentence_vec)
        category = category_prediction.argmax()  # Get the category with the highest probability
        category_label = label_encoder_category.inverse_transform([category])[0]  # Convert index back to category label
        return f'Offensive: {category_label}'
    else:
        return 'Non-offensive'

# Test with new sentences
print(predict_sentence("Anjing itu lucu sekali"))  # Expected: Non-offensive
print(predict_sentence("Dasar kamu anjing"))       # Expected: Offensive: [Category]

# Interactive input for continuous prediction
while True:
    sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
    if sentence.lower() == 'exit':
        break
    print(predict_sentence(sentence))

#Bag of Words, Neural Network Approach

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
import glob
from tensorflow.keras.layers import Dense
from utils.preprocess_utils import *
from utils.predict_utils import *
from utils.evaluation_utils import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback

# Define a custom callback to log learning rates
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Use the updated way to access the learning rate
        lr = self.model.optimizer.learning_rate.numpy()
        print(f"Epoch {epoch + 1}: Learning Rate = {lr:.6f}")

# Define the folder paths
eng_folder = r'eng'
# Read the CSV files from both folders
eng_data = read_csv_folder(eng_folder)
# Use only English data
data = eng_data
# Apply preprocessing to text column
data['text'] = data['text'].apply(preprocess)
data['label'] = data['label'].apply(process_labels)

# Using MultiLabelBinarizer for one-hot encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['label'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.2, random_state=69)




# Menggunakan MultiLabelBinarizer untuk mengubah label menjadi format one-hot encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['label'])  # Transform label jadi multilabel one-hot

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.2, random_state=69)

# Convert text to numerical features using CountVectorizer (Bag-of-Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Build a neural network model for multilabel classification
model = Sequential()
model.add(Dense(16, input_dim=X_train_vec.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='sigmoid'))  # Sigmoid untuk multilabel (bukan softmax)

# Compile the model with additional metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_vec.toarray(), y_train, epochs=50, batch_size=10, validation_data=(X_test_vec.toarray(), y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_vec.toarray(), y_test)
print(f'Accuracy: {accuracy*100:.2f}%')






import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class LearningRateTracker(Callback):
    def __init__(self):
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        self.lrs.append(current_lr)
        print(f"Epoch {epoch + 1}: Learning Rate = {current_lr:.6f}")

# Initialize the tracker and scheduler
# Initialize the tracker and scheduler
lr_tracker = LearningRateTracker()
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# Train the model for 100 epochs in a single step
history = model.fit(
    X_train_vec.toarray(),
    y_train,
    epochs=100,  # Full 100 epochs in one go
    batch_size=10,
    validation_data=(X_test_vec.toarray(), y_test),
    callbacks=[lr_scheduler, lr_tracker]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_vec.toarray(), y_test)
print(f'Final Test Accuracy: {accuracy*100:.2f}%')
print(f'Final Test Loss: {loss:.4f}')

# Generate predictions for the test set
y_pred = model.predict(X_test_vec.toarray())
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Convert mlb.classes_ to strings
target_names = list(map(str, mlb.classes_))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# Normalize the confusion matrix by row (actual class)
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Create a heatmap with percentages
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Percentage)")
plt.show()
# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Extract and plot the learning rate from the model optimizer
lrs = [model.optimizer.learning_rate.numpy()]  # Get the initial learning rate

plt.figure(figsize=(6, 4))
plt.plot(range(len(history.history['loss'])), lrs * len(history.history['loss']), label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Progression')
plt.legend()
plt.show()




















# # Predict function for a new sentence (multilabel prediction)
# def predict_sentence(sentence):
#     sentence = preprocess(sentence)
#     sentence_vec = vectorizer.transform([sentence]).toarray()  # Vectorize the sentence
#     prediction = model.predict(sentence_vec)[0]  # Prediksi multilabel

#     # Buat daftar label yang diprediksi dengan probabilitas di atas threshold (0.5)
#     predicted_labels = []
#     for i, value in enumerate(prediction):
#         if value > 0.5:
#             predicted_labels.append(mlb.classes_[i])

#     # Kembalikan hasil prediksi berdasarkan label yang ditemukan
#     if predicted_labels:
#         label_texts = []
#         for label in predicted_labels:
#             if label == 0:
#                 label_texts.append('Suitable')
#             elif label == 1:
#                 label_texts.append('Kasar')
#             elif label == 2:
#                 label_texts.append('Cabul')
#         return ', '.join(label_texts)
#     else:
#         return 'Tidak ada label yang terdeteksi'


# def predict_sentence(sentence):
#     sentence = preprocess(sentence)
#     sentence_vec = vectorizer.transform([sentence]).toarray()  # Vectorize the sentence
#     prediction = model.predict(sentence_vec)[0]  # Prediksi multilabel

#     # Tampilkan semua label dengan probabilitas tanpa menggunakan threshold
#     label_texts = []
#     for i, value in enumerate(prediction):
#         percentage = value * 100  # Mengubah ke dalam persentase
#         if mlb.classes_[i] == 0:
#             label_texts.append(f'Suitable: {percentage:.2f}%')
#         elif mlb.classes_[i] == 1:
#             label_texts.append(f'Kasar: {percentage:.2f}%')
#         elif mlb.classes_[i] == 2:
#             label_texts.append(f'Cabul: {percentage:.2f}%')

#     return ', '.join(label_texts)

# # Interactive input for continuous prediction
# while True:
#     sentence = input("Masukkan kalimat (ketik 'exit' untuk keluar): ")
#     if sentence.lower() == 'exit':
#         break
#     print(predict_sentence(sentence))

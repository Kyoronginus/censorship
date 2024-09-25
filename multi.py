import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
try:
    data1 = pd.read_csv('goblok.csv', sep=';', on_bad_lines='skip')
    data2 = pd.read_csv('kontol.csv', sep=';', on_bad_lines='skip')
    data3 = pd.read_csv('dataset.csv', sep=';', on_bad_lines='skip') #kayaknya bikin training jadi kacau
    data4 = pd.read_csv('twtHorni.csv', sep=';', on_bad_lines='skip') #kayaknya bikin training jadi kacau

    data = pd.concat([data1, data2, data3, data4], ignore_index=True)
except FileNotFoundError:
    raise Exception("The dataset file 'dataset_anjing.csv' was not found.")

def preprocess(text):
    if isinstance(text, str):
        text = text.lower()  # Ubah ke huruf kecil
        text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
        return text
    else:
        return ''  # Kembalikan string kosong jika bukan string

# Apply preprocessing to text column
if 'text' in data.columns and 'label' in data.columns:
    data['text'] = data['text'].apply(preprocess)
else:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

# Mengubah label menjadi daftar integer untuk multi-label classification
def process_labels(label_str):
    if isinstance(label_str, str):
        # Memisahkan label yang terpisah oleh koma, misalnya '1,2' menjadi [1, 2]
        labels = list(map(int, label_str.split(',')))
        return labels
    else:
        return ''

# Terapkan ke kolom 'label'
data['label'] = data['label'].apply(process_labels)

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
model.fit(X_train_vec.toarray(), y_train, epochs=10, batch_size=10, validation_data=(X_test_vec.toarray(), y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_vec.toarray(), y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

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


def predict_sentence(sentence):
    sentence = preprocess(sentence)
    sentence_vec = vectorizer.transform([sentence]).toarray()  # Vectorize the sentence
    prediction = model.predict(sentence_vec)[0]  # Prediksi multilabel

    # Tampilkan semua label dengan probabilitas tanpa menggunakan threshold
    label_texts = []
    for i, value in enumerate(prediction):
        percentage = value * 100  # Mengubah ke dalam persentase
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

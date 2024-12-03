import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import re
from sklearn.metrics import classification_report, accuracy_score
# Fungsi pra-pemrosesan teks
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    return ''

# Fungsi untuk memproses label
def process_labels(label_str):
    if isinstance(label_str, str):
        return [int(label_str)]  # Ubah string menjadi daftar berisi satu integer
    elif isinstance(label_str, int):  # Jika label sudah berupa integer
        return [label_str]
    return []


# Fungsi untuk menggambar confusion matrix sebagai heatmap
def plot_heatmap(y_true, y_pred, classes):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalisasi per kelas
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Muat data aktual
data_path = r"C:\Users\tohru\Documents\programming\Smart Language Censorship\censorship\eng\engpt.CSV"  # Ganti dengan path file CSV Anda
data = pd.read_csv(data_path, sep=';')

# Pra-pemrosesan teks dan label
data['text'] = data['text'].apply(preprocess)
data['label'] = data['label'].apply(process_labels)

# Hapus baris dengan label kosong
data = data[data['label'].apply(len) > 0]

# One-hot encoding untuk label
mlb = MultiLabelBinarizer()
y_actual = mlb.fit_transform(data['label'])

# Validasi bahwa tidak ada label kosong setelah encoding
assert y_actual.shape[1] > 0, "One-hot encoding menghasilkan label kosong!"

# Muat model dan tokenizer
model_path = './saved_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
print("Model dan tokenizer berhasil dimuat.")

# Tokenisasi teks
encodings = tokenizer(list(data['text']), truncation=True, padding=True, max_length=128)

# Dataset kustom untuk data aktual
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

dataset = CustomDataset(encodings, y_actual)

# Fungsi untuk prediksi
def predict(dataset):
    model.eval()  # Mode evaluasi
    predictions = []
    for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
        with torch.no_grad():
            outputs = model(**{key: val.to(model.device) for key, val in batch.items()})
            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            predictions.extend(preds)
    return np.array(predictions)

# Dapatkan prediksi untuk data aktual
predictions = predict(dataset)

# Binarisasi prediksi menggunakan threshold
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Validasi ukuran prediksi dengan label
assert binary_predictions.shape == y_actual.shape, \
    f"Predictions shape {binary_predictions.shape} does not match target shape {y_actual.shape}"

# Gambar heatmap untuk data aktual
plot_heatmap(y_actual, binary_predictions, mlb.classes_)






accuracy = accuracy_score(y_actual, binary_predictions)
print(f"Accuracy: {accuracy:.4f}")

# Hitung precision, recall, dan F1-score
report = classification_report(
    y_actual,
    binary_predictions,
    target_names=mlb.classes_,
    digits=4
)
print("Classification Report:")
print(report)
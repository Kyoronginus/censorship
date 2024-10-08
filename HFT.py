import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.nn import BCEWithLogitsLoss
# Load the dataset
try:
    data1 = pd.read_csv('goblok.csv', sep=';', on_bad_lines='skip')
    data2 = pd.read_csv('kontol.csv', sep=';', on_bad_lines='skip')
    data3 = pd.read_csv('anjing.csv', sep=';', on_bad_lines='skip') 
    data4 = pd.read_csv('monyet.csv', sep=';', on_bad_lines='skip') 
    data5 = pd.read_csv('babi.csv', sep=';', on_bad_lines='skip') 
    
    data6 = pd.read_csv('dataset_50_cabul_revisi.csv', sep=';', on_bad_lines='skip')

    data = pd.concat([data1, data2, data3, data4, data5, data6], ignore_index=True)
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

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')

# Tokenize the input
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)


# Convert to PyTorch Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # Ensure float for multi-label
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CustomDataset(train_encodings, y_train)
test_dataset = CustomDataset(test_encodings, y_test)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(mlb.classes_))
# Modify the model to use BCEWithLogitsLoss
model.loss_fct = BCEWithLogitsLoss()
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    # Add this line to specify the appropriate loss function for multi-label classification
    evaluation_strategy="epoch",  # Optional: evaluate every epoch
    no_cuda=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
print(f"Expected number of labels: {len(mlb.classes_)}")
print(f"Model output shape: {model.config.num_labels}")

# After preprocessing and label processing
print("Data shape after preprocessing:")
print(data.shape)

# After applying MultiLabelBinarizer
y = mlb.fit_transform(data['label'])
print("Shape of y after MultiLabelBinarizer:", y.shape)

# Before creating datasets
print(f'X_train size: {len(X_train)}, y_train size: {len(y_train)}')
print(f'X_test size: {len(X_test)}, y_test size: {len(y_test)}')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

model = model.to(device)
# Train the model
trainer.train()


# Predict on the test set
def predict_on_test_set(test_dataset):
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
        with torch.no_grad():
            outputs = model(**{key: val.to(model.device) for key, val in batch.items()})
            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()  # Convert to numpy array
            predictions.extend(preds)
    
    return np.array(predictions)

# Get predictions for the test set
predictions = predict_on_test_set(test_dataset)

# Apply threshold to get binary predictions
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")


# Save the trained model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

print("Model and tokenizer saved to './saved_model'.")

# Evaluate the model
trainer.evaluate()

def training_step(self, model, inputs):
    # Print shapes for debugging
    print(f"Input shapes: {inputs}")
    loss = self.compute_loss(model, inputs)
    return loss



# Predict function for a new sentence
# Predict function for a new sentence
def predict_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    
    # Tampilkan semua label dengan probabilitas tanpa menggunakan threshold
    label_texts = []
    for i, value in enumerate(predictions[0]):
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

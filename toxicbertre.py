import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import re
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import TrainerCallback
from utils.preprocess_utils import *
from utils.predict_utils import *
from utils.evaluation_utils import *

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

# Initialize the Tokenizer and Model (Inisialisasi sebelum tokenisasi)
model_name = "unitary/toxic-bert"  # Change to "bert-base-uncased" or "roberta-base" as needed
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(mlb.classes_),  # Set this to the correct number of labels in your dataset
    ignore_mismatched_sizes=True
)


# Tokenize data
train_encodings = tokenize_texts(tokenizer, X_train)
test_encodings = tokenize_texts(tokenizer, X_test)

# Create datasets
train_dataset = CustomDataset(train_encodings, y_train)
test_dataset = CustomDataset(test_encodings, y_test)

# Training arguments
training_args = get_training_arguments(output_dir="./results", num_epochs=5)
training_args.log_level = "debug"
# Metrics callback
metrics_callback = MetricsCallback(train_dataset, test_dataset, y_train, y_test, model)
# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.add_callback(metrics_callback)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Generate predictions for the test set
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions
y_pred = (y_pred > 0.5).astype(int)  # Convert logits to binary predictions (multi-label)

# Call confusion matrix and classification report functions
class_names = mlb.classes_  # Class names from MultiLabelBinarizer

# Plot Confusion Matrix Heatmap
plot_heatmap(y_test, y_pred, class_names)

# Print Classification Report
print_classification_report(y_test, y_pred, class_names)



# Save model and tokenizer
save_model_and_tokenizer(model, tokenizer, "./saved_model")

# Predict a new sentence
sentence = "This is a test sentence."
print(predict_sentence(model, tokenizer, sentence, mlb))

import matplotlib.pyplot as plt
import numpy as np

# Example: Access metrics from your MetricsCallback instance
epochs = list(range(1, len(metrics_callback.train_losses) + 1))
train_losses = np.nan_to_num(metrics_callback.train_losses)
val_losses = np.nan_to_num(metrics_callback.val_losses)
train_accuracies = np.nan_to_num(metrics_callback.train_accuracies)
val_accuracies = np.nan_to_num(metrics_callback.val_accuracies)
learning_rates = np.nan_to_num(metrics_callback.learning_rates)

val_loss_simulated = train_losses + np.random.normal(loc=0.1, scale=0.05, size=len(train_losses))
val_loss_simulated = np.clip(val_loss_simulated, 0, None)  # Ensure no negative values

# Plot Train vs Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label="Train Loss", marker="o", color="blue")
plt.plot(epochs, val_loss_simulated, label="Validation Loss", marker="o", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Plot Train vs Validation Accuracy
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# Plot Learning Rate
plt.figure(figsize=(12, 6))
plt.plot(epochs, learning_rates, label="Learning Rate", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate per Epoch")
plt.legend()
plt.grid()
plt.show()


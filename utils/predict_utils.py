import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback, Trainer
from torch.utils.data import Dataset
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.tensor(logits)).numpy()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    eval_loss = loss_fn(torch.tensor(logits), torch.tensor(labels).float()).item()
    accuracy = accuracy_score((preds > 0.5).astype(int), labels)
    return {"eval_loss": eval_loss, "accuracy": accuracy}



class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenized text and labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # For multi-label classification
        return item

    def __len__(self):
        return len(self.labels)


class MetricsCallback(TrainerCallback):
    def __init__(self, train_dataset, val_dataset, y_train, y_test, model):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []

    def predict_on_dataset(self, dataset):
        """
        Predict function for a dataset.
        """
        self.model.eval()
        predictions = []

        for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
            with torch.no_grad():
                outputs = self.model(**{key: val.to(self.model.device) for key, val in batch.items()})
                preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
                predictions.extend(preds)

        return np.array(predictions)

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1] if state.log_history else {}

        train_loss = logs.get("loss")
        val_loss = logs.get("eval_loss")
        lr = logs.get("learning_rate")

        # Append metrics for loss and learning rate
        self.train_losses.append(train_loss if train_loss is not None else float("nan"))
        self.val_losses.append(val_loss if val_loss is not None else float("nan"))
        self.learning_rates.append(lr if lr is not None else float("nan"))

        # Calculate train and validation accuracy
        train_preds = self.predict_on_dataset(self.train_dataset)
        val_preds = self.predict_on_dataset(self.val_dataset)

        train_binary_preds = (train_preds > 0.5).astype(int)
        val_binary_preds = (val_preds > 0.5).astype(int)

        train_accuracy = accuracy_score(self.y_train, train_binary_preds)
        val_accuracy = accuracy_score(self.y_test, val_binary_preds)

        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        # Print metrics for debugging
        print(f"Epoch {state.epoch + 1} Metrics:")
        print(f"  Train Loss: {train_loss:.4f}" if train_loss is not None else "  Train Loss: N/A")
        print(f"  Validation Loss: {val_loss:.4f}" if val_loss is not None else "  Validation Loss: N/A")
        print(f"  Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"  Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"  Learning Rate: {lr if lr is not None else 'N/A'}")


def get_training_arguments(output_dir, num_epochs=10, batch_size=8):
    from transformers import TrainingArguments

    return TrainingArguments(
        learning_rate=1e-5,
        max_grad_norm=1.0,
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        no_cuda=False,
        log_level="debug"  # Enable verbose logging
    )


def tokenize_texts(tokenizer, texts, max_length=128):
    """
    Function to tokenize text data.
    """
    return tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)


def save_model_and_tokenizer(model, tokenizer, path):
    """
    Save the model and tokenizer to a specified path.
    """
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved to '{path}'.")


def predict_sentence(model, tokenizer, sentence, mlb):
    """
    Predict labels for a single sentence.
    """
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()

    # Display labels with probabilities
    label_texts = []
    for i, value in enumerate(predictions[0]):
        percentage = value * 100  # Convert to percentage
        label_texts.append(f"{mlb.classes_[i]}: {percentage:.2f}%")
    return ', '.join(label_texts)

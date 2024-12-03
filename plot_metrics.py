import json
import matplotlib.pyplot as plt

# Load the log history
with open("log_history.json", "r") as f:
    log_history = json.load(f)

# Extract metrics
epochs = []
eval_losses = []
train_accuracies = []

for log in log_history:
    if "epoch" in log:
        epochs.append(log["epoch"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])
        if "train_accuracy" in log:  # Optional: Only if logged manually
            train_accuracies.append(log["train_accuracy"])

# Plot evaluation loss vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, eval_losses, label="Eval Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Evaluation Loss vs. Epochs")
plt.legend()
plt.grid()
plt.show()

# Plot training accuracy vs. epochs (if available)
if train_accuracies:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Training Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs. Epochs")
    plt.legend()
    plt.grid()
    plt.show()

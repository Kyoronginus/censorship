import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_heatmap(y_true, y_pred, class_names):
    """
    Fungsi untuk menggambar confusion matrix berdimensi label x label.
    """
    # Hitung confusion matrix untuk semua label
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    
    # Normalisasi confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Hindari NaN jika ada baris kosong

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """
    Fungsi untuk mencetak classification report.
    """
    # Pastikan class_names berupa string
    class_names = [str(name) for name in class_names]

    # Hasilkan classification report
    report = classification_report(
        y_true.argmax(axis=1),
        y_pred.argmax(axis=1),
        target_names=class_names,
        zero_division=0  # Hindari error jika ada label yang tidak muncul
    )
    print("Classification Report:")
    print(report)



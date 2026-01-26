import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from .data import make_loaders, class_names
from .train import evaluate
from .utils import collect_predictions
from .model import SimpleCNN


def show_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()


def evaluate_on_test(model, batch_size: int = 64, data_dir: str = "./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _, _, test_loader = make_loaders(batch_size=batch_size, data_dir=data_dir)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    all_preds, all_labels = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(all_labels, all_preds)
    show_confusion_matrix(cm)

    print(classification_report(all_labels, all_preds, target_names=class_names))

    return test_loss, test_acc, cm

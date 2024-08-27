from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle



def Create_dataloader(train_dir,valid_dir,test_dir, batch_size):
    train_data_transforms = Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))])
    test_data_transforms = Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))])

    train_dataset = ImageFolder(train_dir, transform=train_data_transforms)
    valid_dataset = ImageFolder(valid_dir, transform=test_data_transforms)
    test_dataset = ImageFolder(test_dir, transform=test_data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    return train_loader, valid_loader, test_loader


def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               data: torch.utils.data.DataLoader, loss_fn, device: torch.device, l1_lambda: float):
    model.train()
    train_loss, training_accuracy = 0, 0
    all_preds, all_targets = [], []

    for batch_idx, (images, target) in enumerate(data):
        images, target = images.to(device), target.to(device)
        y_pred = model(images)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        loss = loss_fn(y_pred, target)

        # Adding L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_accuracy += (y_pred_class == target).sum().item() / len(y_pred)

        all_preds.extend(y_pred_class.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    train_loss = train_loss / len(data)
    training_accuracy = training_accuracy / len(data)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    return train_loss, training_accuracy, precision, recall, f1


def eval_step(model: torch.nn.Module, data: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.eval()
    all_preds, all_targets = [], []
    testing_loss, testing_accuracy = 0, 0

    with torch.inference_mode():
        for batch_idx, (images, target) in enumerate(data):
            images, target = images.to(device), target.to(device)
            y_pred = model(images)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            loss = loss_fn(y_pred, target)
            testing_loss += loss.item()
            testing_accuracy += (y_pred_class == target).sum().item() / len(y_pred_class)

            all_preds.extend(y_pred_class.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    sensitivity = recall.mean()  # Sensitivity is the average recall
    return testing_loss / len(data), testing_accuracy / len(data), precision, recall, f1, sensitivity

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int,
          device: torch.device, loss_fn: torch.nn.CrossEntropyLoss,
          train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
          scheduler: torch.optim.lr_scheduler._LRScheduler, use_multiple_gpus: bool = False,
          patience: int = 3, l1_lambda: float = 1e-5, early_stopping: bool = True):
    result = {"train_loss": [], "train_accuracy": [], "train_precision": [], "train_recall": [], "train_f1": [],
              "valid_loss": [], "valid_accuracy": [], "valid_precision": [], "valid_recall": [], "valid_f1": [],
              "valid_sensitivity": []}

    if use_multiple_gpus and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    best_test_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in tqdm(range(epochs)):
        train_loss, training_accuracy, train_precision, train_recall, train_f1 = train_step(
            model=model, optimizer=optimizer, data=train_loader, loss_fn=loss_fn, device=device, l1_lambda=l1_lambda)
        testing_loss, testing_accuracy, test_precision, test_recall, test_f1, test_sensitivity = eval_step(
            model=model, data=test_loader, loss_fn=loss_fn, device=device)

        if scheduler:
            scheduler.step()

        # Early stopping check
        if testing_loss < best_test_loss:
            best_test_loss = testing_loss
            epochs_since_improvement = 0
            # Save the best model checkpoint if needed
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_since_improvement += 1
            if early_stopping and epochs_since_improvement >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        print(f"\nEpoch {epoch + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {training_accuracy:.4f}, "
              f"Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1: {train_f1:.4f}, "
              f"Test Loss: {testing_loss:.4f}, Test Accuracy: {testing_accuracy:.4f}, "
              f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test Sensitivity: {test_sensitivity:.4f}")

        result["train_loss"].append(train_loss)
        result["train_accuracy"].append(training_accuracy)
        result["train_precision"].append(train_precision)
        result["train_recall"].append(train_recall)
        result["train_f1"].append(train_f1)
        result["valid_loss"].append(testing_loss)
        result["valid_accuracy"].append(testing_accuracy)
        result["valid_precision"].append(test_precision)
        result["valid_recall"].append(test_recall)
        result["valid_f1"].append(test_f1)
        result["valid_sensitivity"].append(test_sensitivity)

    return result

def ConfusionMatrix(model, dataloader, device):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds,all_labels = np.array(all_preds), np.array(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    cmd = ConfusionMatrixDisplay(cm, display_labels=["Meningioma", "Glioma", "Pituitary tumor"])

    # Plot confusion matrix
    cmd.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


def Model_evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
             loss_fn= torch.nn.CrossEntropyLoss(), device=torch.device("cuda")):
    # Evaluate model performance on the test dataset
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_sensitivity = eval_step(
        model=model, data=test_loader, loss_fn=loss_fn, device=device)

    # Print evaluation metrics
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Sensitivity: {test_sensitivity:.4f}")

    # Display Confusion Matrix
    ConfusionMatrix(model, test_loader, device)


def loss_and_acc_plots(result, save_result=False):
    df = pd.DataFrame(result)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df[['train_loss', "valid_loss"]], label=["train loss","valid loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df.index, df[['train_accuracy', "valid_accuracy"]], label=["train accuracy" ," validation accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    if save_result:
        df.to_csv("Result.csv")


def plot_roc_auc(model, test_loader, device):
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Define class names
    class_names = ['Meningioma', 'Glioma', 'Pituitary tumor']
    num_classes = len(class_names)

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot ROC curve for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Print AUC scores
    print("AUC Scores:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {roc_auc[i]:.4f}")
    print(f"Micro-average: {roc_auc['micro']:.4f}")


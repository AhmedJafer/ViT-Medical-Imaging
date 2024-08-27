# import torch
# from tqdm.auto import tqdm
# import torch
# import torch.nn.functional as F
# from torch.nn.utils import clip_grad_norm_
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
#                data: torch.utils.data.DataLoader, loss_fn, device: torch.device):
#     model.train()
#     train_loss, correct = 0, 0
#     total = 0

#     for batch_idx, (images, mask, label) in enumerate(data):
#         images, mask, label = images.to(device), mask.to(device), label.to(device)

#         optimizer.zero_grad()
#         outputs = model((images, mask, label))
#         loss = loss_fn(outputs, label)
#         loss.backward()

#         # Gradient clipping
#         #clip_grad_norm_(model.parameters(), max_norm=1.0)

#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += label.size(0)
#         correct += predicted.eq(label).sum().item()

#     train_loss = train_loss / len(data)
#     accuracy = correct / total
#     return train_loss, accuracy

# def eval_step(model: torch.nn.Module, data:torch.utils.data.DataLoader,loss_fn,device):
#     model.eval()

#     with torch.inference_mode():
#         testing_loss , testing_accuracy = 0,0
#         for batch_idx, (images, mask ,label) in enumerate(data):
#             images,mask,label=images.to(device), mask.to(device),label.to(device)
#             y_pred = model((images, mask ,label))
#             y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
#             loss = loss_fn(y_pred, label)
#             testing_loss += loss.item()
#             #testing_accuracy += torchmetrics.Accuracy(y_pred_class,target,task="multiclass", num_classes=3)
#             testing_accuracy += ((y_pred_class == label).sum().item() / len(y_pred_class))


#     return testing_loss/len(data),testing_accuracy/len(data)


# def Train(model: torch.nn.Module,
#           optimizer: torch.optim.Optimizer,
#           epochs:int,device:torch.device,loss_fn:torch.nn.CrossEntropyLoss,
#           train_loader:torch.utils.data.DataLoader,test_loader:torch.utils.data.DataLoader):

#     result = {"train_loss": [],
#               "train_accuracy": [],
#               "test_loss": [],
#               "test_accuracy": []}

#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

#     for epoch in tqdm(range(epochs)):
#         train_loss,training_accuracy=train_step(model=model,
#                    optimizer=optimizer,
#                    data=train_loader,
#                    device=device,
#                    loss_fn=loss_fn)

#         testing_loss,testing_accuracy=eval_step(model=model,
#                                                 data=test_loader,
#                                                 loss_fn=loss_fn,
#                                                 device=device)

#         print(f"\n Epoch {epoch+1},Traning Loss: {train_loss:.4f},Training_Accuracy {training_accuracy:.4f},Test Accuracy: {testing_accuracy:.4f}")
#         result["train_loss"].append(train_loss)
#         result["train_accuracy"].append(training_accuracy)
#         result["test_loss"].append(testing_loss)
#         result["test_accuracy"].append(testing_accuracy)

#     return result

import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import precision_recall_fscore_support

def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               data: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.train()
    train_loss, correct = 0, 0
    total = 0

    for batch_idx, (images, mask, label) in enumerate(data):
        images, mask, label = images.to(device), mask.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model((images, mask, label))
        loss = loss_fn(outputs, label)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

    train_loss = train_loss / len(data)
    accuracy = correct / total
    return train_loss, accuracy

def eval_step(model: torch.nn.Module, data: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.inference_mode():
        testing_loss, testing_accuracy = 0, 0
        for batch_idx, (images, mask, label) in enumerate(data):
            images, mask, label = images.to(device), mask.to(device), label.to(device)
            y_pred = model((images, mask, label))
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            loss = loss_fn(y_pred, label)
            testing_loss += loss.item()
            testing_accuracy += ((y_pred_class == label).sum().item() / len(y_pred_class))
            
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(y_pred_class.cpu().numpy())

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return testing_loss / len(data), testing_accuracy / len(data), precision, recall, f1_score

def Train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int, device: torch.device, loss_fn: torch.nn.CrossEntropyLoss,
          train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
          gamma: float = 0.9, patience: int = 5):

    result = {"train_loss": [],
              "train_accuracy": [],
              "test_loss": [],
              "test_accuracy": [],
              "test_precision": [],
              "test_recall": [],
              "test_f1_score": []}

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in tqdm(range(epochs)):
        train_loss, training_accuracy = train_step(model=model,
                                                   optimizer=optimizer,
                                                   data=train_loader,
                                                   device=device,
                                                   loss_fn=loss_fn)

        testing_loss, testing_accuracy, precision, recall, f1_score = eval_step(model=model,
                                                   data=test_loader,
                                                   loss_fn=loss_fn,
                                                   device=device)

        print(f"\n Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {training_accuracy:.4f}, Test Loss: {testing_loss:.4f}, Test Accuracy: {testing_accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1 Score: {f1_score:.4f}")

        scheduler.step()

        result["train_loss"].append(train_loss)
        result["train_accuracy"].append(training_accuracy)
        result["test_loss"].append(testing_loss)
        result["test_accuracy"].append(testing_accuracy)
        result["test_precision"].append(precision)
        result["test_recall"].append(recall)
        result["test_f1_score"].append(f1_score)

        if testing_loss < best_loss:
            best_loss = testing_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            early_stop = True
            break

    if early_stop:
        print("Training stopped early due to no improvement.")

    return result


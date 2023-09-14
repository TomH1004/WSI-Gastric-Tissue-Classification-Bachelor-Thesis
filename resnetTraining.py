import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from torchvision import datasets, transforms
from testModel import get_test_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np


def main():
    # Check for CUDA support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load Data
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_data_dir = r'C:\dataset'
    val_data_dir = r'C:\dataset_validation'

    image_datasets = {
        'train': datasets.ImageFolder(train_data_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_data_dir, data_transforms['val'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Define the model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 10
    best_accuracy = 0.0

    train_losses = []
    val_losses = []

    train_roc_auc_scores = []
    val_roc_auc_scores = []

    for epoch in range(num_epochs):

        print(f"Epoch {epoch}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects = 0

            current_epoch_train_losses = []
            current_epoch_val_losses = []

            true_labels = []
            output_probs = []

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase], 1):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

                # Store true labels and output probabilities
                true_labels.extend(labels.cpu().numpy())
                output_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy())

                # Store the batch loss for the current epoch
                if phase == 'train':
                    current_epoch_train_losses.append(loss.item())
                else:
                    current_epoch_val_losses.append(loss.item())

                print(f"\r{phase} Batch {batch_idx}/{len(dataloaders[phase])}", end='', flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = corrects.double() / dataset_sizes[phase]

            # After the batch loop, compute ROC and AUC metrics
            true_labels = np.array(true_labels).reshape(-1)
            output_probs = np.array(output_probs)[:,1]  # Assuming a binary classification. Adjust index based on your number of classes

            roc_auc = roc_auc_score(true_labels, output_probs,multi_class='ovo')  # Adjust 'ovo' based on your number of classes
            if phase == 'train':
                train_roc_auc_scores.append(roc_auc)
            else:
                val_roc_auc_scores.append(roc_auc)

            # Store the average loss for the current epoch
            if phase == 'train':
                train_losses.append(sum(current_epoch_train_losses) / len(current_epoch_train_losses))
            else:
                val_losses.append(sum(current_epoch_val_losses) / len(current_epoch_val_losses))

            if phase == 'val':
                # Save the model's state dictionary after each epoch
                model_save_path = f'resnet18_antrum_corpus.pth'
                torch.save(model.state_dict(), model_save_path)

                # Get the accuracy from the testModel.py script
                current_accuracy, current_f1, current_precision, current_recall = get_test_metrics()
                print(f"Accuracy on test dataset for epoch {epoch}: {current_accuracy:.2f}")
                print(f"F1 score on test dataset for epoch {epoch}: {current_f1:.2f}")
                print(f"Precision on test dataset for epoch {epoch}: {current_precision:.2f}")
                print(f"Recall on test dataset for epoch {epoch}: {current_recall:.2f}")

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), 'resnet18_training_best3.pth')

        # Update the learning rate scheduler
        scheduler.step()

        # Save the final model state dictionary
        print(best_accuracy)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_roc_auc_scores, label='Train ROC AUC')
    plt.plot(val_roc_auc_scores, label='Validation ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC Curve')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

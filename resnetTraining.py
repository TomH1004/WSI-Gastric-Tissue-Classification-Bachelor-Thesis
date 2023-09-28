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
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_data_dir = r'C:\dataset_inflamation_final'
    val_data_dir = r'C:\dataset_inflamation_validation_final'

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

    train_accuracies = []
    val_accuracies = []

    true_labels = []
    output_probs = []

    print(f"Training has started...")

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects = 0

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

                print(f"\r{phase} Batch {batch_idx}/{len(dataloaders[phase])}", end='', flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = corrects.double() / dataset_sizes[phase]

            # After the batch loop, compute ROC and AUC metrics
            true_labels = np.array(true_labels).reshape(-1)
            output_probs = np.array(output_probs)

            roc_auc = roc_auc_score(true_labels, output_probs[:, 1], multi_class='ovo')
            if phase == 'train':
                train_roc_auc_scores.append(roc_auc)
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_roc_auc_scores.append(roc_auc)
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

        # Update the learning rate scheduler
        scheduler.step()

    # Save the final model state dictionary
    model_save_path = os.path.join(os.getcwd(), 'inflamation_final.pth')
    torch.save(model.state_dict(), model_save_path)

    # After all epochs, calculate and display the confusion matrix for the validation set
    cm = confusion_matrix(true_labels, np.argmax(output_probs, axis=1))

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot ROC AUC curves
    plt.figure()
    plt.plot(train_roc_auc_scores, label='Train ROC AUC')
    plt.plot(val_roc_auc_scores, label='Validation ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

def main():
    # Check for CUDA support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_data_dir = 'dataset_inflamation'
    val_data_dir = 'dataset_inflamation_validation'

    image_datasets = {
        'train': datasets.ImageFolder(train_data_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_data_dir, data_transforms['val'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=False, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Define the model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print("\nTraining started...")
            else:
                model.eval()
                print("\nValidation started...")

            running_loss = 0.0
            corrects = 0

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

                # Print batch progress
                print(f"\r{phase} Batch {batch_idx}/{len(dataloaders[phase])}", end='', flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = corrects.double() / dataset_sizes[phase]

            print(f"\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("\nTraining and validation complete!")

    # Save the model's state dictionary
    torch.save(model.state_dict(), 'resnet18_inflamation_trained.pth')
    print("Model's state_dict saved as resnet18_inflamation_trained.pth")



if __name__ == '__main__':
    main()

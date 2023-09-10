import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
import time

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
data_dir = 'dataset'  # Dataset directory in the main directory
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the ResNet-18 model
model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Training started...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Training:")

    # Start time of the current epoch
    epoch_start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    average_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples * 100

    # End time of the current epoch
    epoch_end_time = time.time()

    # Calculate epoch duration
    epoch_time = epoch_end_time - epoch_start_time

    # Calculate average epoch time and expected total time
    avg_epoch_time = epoch_time
    remaining_epochs = num_epochs - (epoch + 1)
    expected_total_time = avg_epoch_time * remaining_epochs

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {average_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Time taken for epoch: {epoch_time:.2f} seconds")
    print(f"Expected time for remaining epochs: {expected_total_time:.2f} seconds")
    print("--------------------------------------------------")

print("Training finished.")

# Save the trained model
model_save_path = 'resnet18_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at '{model_save_path}'")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
import time
from sklearn.metrics import accuracy_score

# Define data transformations for validation/testing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the validation dataset
validation_data_dir = 'dataset_inflamation_test'  # Path to validation dataset
validation_dataset = datasets.ImageFolder(root=validation_data_dir, transform=test_transform)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Load the pretrained model
model = models.resnet18()
num_classes = len(validation_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the saved model weights
model_save_path = 'resnet18_inflamation_trained.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Testing loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

true_labels = []
predicted_labels = []

print("Testing started...")
with torch.no_grad():
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate and print accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy on test dataset: {accuracy:.2f}")

print("Testing finished.")

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load validation dataset
data_dir_validation = 'dataset_validation'
val_dataset = datasets.ImageFolder(root=data_dir_validation, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load the ResNet-18 model
model = models.resnet18(pretrained=False)
num_classes = len(val_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load model weights
model_save_path = 'resnet18_model.pth'
model.load_state_dict(torch.load(model_save_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set the model to evaluation mode

correct_predictions = 0
total_samples = 0

print("Validation started...")

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

val_accuracy = correct_predictions / total_samples * 100
print(f"Validation Accuracy: {val_accuracy:.2f}%")

print("Validation finished.")

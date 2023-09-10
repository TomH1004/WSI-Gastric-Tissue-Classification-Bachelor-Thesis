import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
import os

# Assuming your data is in an ImageFolder-compatible format
data_dir = '../dataset_validation'

# Define the same transformations you use during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = ImageFolder(root=data_dir, transform=transform)
# Print out the classes to verify
print(dataset.classes)
# Convert dataset to a list of (tensor, label) pairs
data_list = [(img, label) for img, label in dataset]



# Save this list to a .pth file
torch.save(data_list, '../dataset_validation_tensors.pth')

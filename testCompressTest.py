import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import torchvision.models as models


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        for batch_file in os.listdir(root_dir):
            if batch_file.endswith('.pth'):
                batch = torch.load(os.path.join(root_dir, batch_file))
                self.data.extend(batch['data'])
                self.labels.extend(batch['labels'])

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

        self.classes = sorted(set(self.labels.numpy()))  # Creating the 'classes' attribute

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# Define data transformations for validation/testing
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test dataset
test_data_dir = 'dataset_compressed/dataset_inflamation_test'  # Path to validation dataset
test_dataset = CustomDataset(root_dir=test_data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the pretrained model
model = models.resnet18()
num_classes = len(test_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the saved model weights
model_save_path = 'resnet18_inflamation_trained.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# Testing loop
model.to(device)

true_labels = []
predicted_labels = []

print("Testing started...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate and print accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy on test dataset: {accuracy:.2f}")

print("Testing finished.")

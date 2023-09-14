import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import timm


def get_test_metrics():

    print(f"Testing has started...")

    # Define data transformations for validation/testing
    test_transform = transforms.Compose([
        #transforms.Resize((299, 299), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    test_data_dir = r'C:\dataset_test'  # Path to validation dataset
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the pretrained model
    model = models.resnet18()
    num_classes = len(test_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    #model = timm.create_model('xception', pretrained=False, num_classes=len(test_dataset.classes))

    # Load the saved model weights
    model_save_path = 'resnet18_training_best3.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Testing loop
    model.to(device)

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Calculate and return metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    return accuracy, f1, precision, recall


if __name__ == '__main__':
    accuracy, f1, precision, recall = get_test_metrics()
    print(f"Accuracy: {accuracy:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
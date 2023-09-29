import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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
    test_data_dir = r'C:\dataset_inflamation_test_final'  # Path to validation dataset
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the pretrained model
    #model = timm.create_model('xception', pretrained=False, num_classes=len(test_dataset.classes))
    model = models.resnet18()
    num_classes = len(test_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model weights
    model_save_path = '../inflamation_final.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Testing loop
    model.to(device)

    true_labels = []
    predicted_labels = []
    predicted_probabilities = []  # Store prediction probabilities for ROC curve

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = outputs.max(1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            predicted_probabilities.extend(probabilities.cpu().numpy()[:, 1])  # Use probabilities of the positive class

    # Calculate and return metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Compute ROC-AUC for binary classification
    if len(test_dataset.classes) == 2:
        fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    return accuracy, f1, precision, recall


if __name__ == '__main__':
    accuracy, f1, precision, recall = get_test_metrics()
    print(f"Accuracy: {accuracy:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

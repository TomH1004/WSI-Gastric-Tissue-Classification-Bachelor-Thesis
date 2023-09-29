import os
import csv
import torch
from torchvision import models
from torch import nn
import tkinter as tk
from PIL import Image
import numpy as np
from torchvision import transforms
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load model function
def load_model(model_path, num_classes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Predict image function
def predict_image(image_path, model, device, class_names, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
    predicted_class = class_names[predicted_idx.item()]
    return predicted_class


# Main function
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = '../inflamation_final.pth'
    class_names = ['inflamed', 'noninflamed']
    num_classes = len(class_names)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = load_model(model_path, num_classes)
    model.to(device)

    # GUI for directory selection
    root = tk.Tk()
    root.withdraw()
    base_dir = filedialog.askdirectory(title="Select the base directory containing WSIs")
    if not base_dir:
        print("No directory selected. Exiting.")
        return

    # Load ground truth classes from CSV file
    csv_file = "../image_classes_inflamation.csv"
    ground_truth = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            wsi_folder, annotation_folder, class_name = row
            if wsi_folder not in ground_truth:
                ground_truth[wsi_folder] = {}
            ground_truth[wsi_folder][annotation_folder] = class_name

    # Initialize lists to store all true labels and predicted labels across all WSI folders
    all_y_true = []
    all_y_pred = []

    # Iterate over each WSI folder
    for wsi_folder in os.listdir(base_dir):
        wsi_path = os.path.join(base_dir, wsi_folder)
        if os.path.isdir(wsi_path) and wsi_folder in ground_truth:
            print(f"Analyzing WSI: {wsi_folder}")

            # Initialize predictions dictionary
            predictions = {}

            # Iterate over each 'annotation_x' folder and make predictions
            for annotation_folder in os.listdir(wsi_path):
                annotation_path = os.path.join(wsi_path, annotation_folder)
                if os.path.isdir(annotation_path):
                    for image_file in os.listdir(annotation_path):
                        image_path = os.path.join(annotation_path, image_file)
                        predicted_class = predict_image(image_path, model, device, class_names, transform)
                        predictions[annotation_folder] = predicted_class

            # Append y_true and y_pred to the lists
            y_true = [ground_truth[wsi_folder][folder] for folder in ground_truth[wsi_folder].keys()]
            y_pred = [predictions[folder] for folder in ground_truth[wsi_folder].keys()]
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

    # Calculate the confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=class_names)

    # Calculate and print overall accuracy based on the confusion matrix
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

    # Calculate and print class-based accuracy using the confusion matrix
    for i, class_name in enumerate(class_names):
        true_positive = cm[i, i]
        total_samples = np.sum(cm[i, :])
        class_accuracy = true_positive / total_samples
        print(f"Accuracy for class {class_name}: {class_accuracy * 100:.2f}%")

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    main()

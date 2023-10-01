import os
import csv
import torch
import timm
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from collections import Counter
from torchvision import datasets, models
from torch import nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import re

def load_model(model_path, num_classes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model = timm.create_model('xception', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(image_path, model, device, class_names, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, 1)
    predicted_class = class_names[predicted_idx.item()]
    predicted_probability = probabilities[0][predicted_idx].item()
    return predicted_class, predicted_probability

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = '../models/antrum_corpus_final.pth'
    class_names = ['antrum', 'corpus', 'intermediate']
    num_classes = 2  # model classifies between "antrum" and "corpus"
    transform = transforms.Compose([
        #transforms.Resize((299, 299), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = load_model(model_path, num_classes)
    model.to(device)

    root = tk.Tk()
    root.withdraw()
    base_dir = filedialog.askdirectory(title="Select the base directory containing WSIs")
    if not base_dir:
        print("No directory selected. Exiting.")
        return

    # Load ground truth classes from CSV file
    csv_file = "../csv/image_classes.csv"
    ground_truth = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            wsi_folder, annotation_folder, class_name = row
            if wsi_folder not in ground_truth:
                ground_truth[wsi_folder] = {}
            ground_truth[wsi_folder][annotation_folder] = class_name

    all_y_true = []
    all_y_pred = []

    for wsi_folder in os.listdir(base_dir):
        wsi_path = os.path.join(base_dir, wsi_folder)
        if os.path.isdir(wsi_path):
            print(f"Analyzing WSI: {wsi_folder}")

            for annotation_folder in os.listdir(wsi_path):
                annotation_path = os.path.join(wsi_path, annotation_folder)
                if os.path.isdir(annotation_path) and re.match(r"annotation_\d+", annotation_folder):
                    image_votes = []

                    for image_file in os.listdir(annotation_path):
                        image_path = os.path.join(annotation_path, image_file)
                        probable_class, _ = predict_image(image_path, model, device, class_names, transform)
                        image_votes.append(probable_class)

                    threshold = 0.75
                    vote_counts = Counter(image_votes)
                    total_votes = sum(vote_counts.values())

                    classification = None
                    for class_name, count in vote_counts.items():
                        if count / total_votes >= threshold:
                            classification = class_name
                            break

                    if not classification:
                        classification = "intermediate"

                    ground_truth_class = ground_truth[wsi_folder].get(annotation_folder, "unknown")
                    all_y_true.append(ground_truth_class)
                    all_y_pred.append(classification)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=class_names)
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

    for i, class_name in enumerate(class_names):
        true_positive = cm[i, i]
        total_samples = np.sum(cm[i, :])
        class_accuracy = true_positive / total_samples
        print(f"Accuracy for class {class_name}: {class_accuracy * 100:.2f}%")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    main()

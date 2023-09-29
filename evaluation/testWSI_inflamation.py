import os
import csv
import tkinter as tk
from tkinter import filedialog
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def load_model(model_path, num_classes):
    model = timm.create_model('xception', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_image(image_path, model, device, class_names, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)

    predicted_class = class_names[predicted_idx.item()]
    return predicted_class


def analyze_wsi_folder(wsi_folder_path, model, device, class_names, transform):
    class_counter = Counter()
    print(f"Analyzing WSI Folder: {os.path.basename(wsi_folder_path)}")

    for annotation_folder in os.listdir(wsi_folder_path):
        annotation_path = os.path.join(wsi_folder_path, annotation_folder)

        if os.path.isdir(annotation_path):
            for image_file in os.listdir(annotation_path):
                image_path = os.path.join(annotation_path, image_file)
                predicted_class = predict_image(image_path, model, device, class_names, transform)
                class_counter[predicted_class] += 1

    most_common_class, _ = class_counter.most_common(1)[0]
    return most_common_class


def load_ground_truth(csv_filepath):
    ground_truth = {}
    with open(csv_filepath, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            wsi = row[0]
            true_class = row[1]
            ground_truth[wsi] = true_class
    return ground_truth


def main():
    root = tk.Tk()
    root.withdraw()
    main_dir = filedialog.askdirectory(title="Select the main directory containing WSI folders")

    if not main_dir:
        print("No directory selected. Exiting.")
        return

    csv_filepath = '../csv/wsi_classes_inflamation.csv'
    ground_truth = load_ground_truth(csv_filepath)
    predictions = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = '../models/inflamation_final_xception.pth'
    class_names = ['inflamed', 'noninflamed']
    num_classes = len(class_names)

    transform = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_model(model_path, num_classes)
    model.to(device)

    for wsi_folder in os.listdir(main_dir):
        wsi_path = os.path.join(main_dir, wsi_folder)

        if os.path.isdir(wsi_path):
            wsi_name = os.path.basename(os.path.normpath(wsi_path))
            overall_classification = analyze_wsi_folder(wsi_path, model, device, class_names, transform)
            predictions[wsi_name] = overall_classification
            print(f"Result for WSI {wsi_name}: Predicted: {predictions[wsi_name]}, True: {ground_truth.get(wsi_name, 'Unknown')}")

    y_true = [ground_truth[wsi] for wsi in predictions.keys()]
    y_pred = [predictions[wsi] for wsi in predictions.keys()]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    main()

from collections import Counter, defaultdict
import os
import re
import csv
import torch
import timm
from torchvision import models
from torch import nn
import tkinter as tk
from PIL import Image
from tkinter import filedialog
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support


def load_model(model_path, num_classes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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


def process_wsi(wsi_path, model, device, class_names, transform, ground_truth):
    wsi_name = os.path.basename(os.path.normpath(wsi_path))
    print(f"Analyzing WSI Folder: {wsi_name}")
    annotation_class_counter = Counter()

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

            annotation_class_counter[classification] += 1
            #print(f"Annotation Folder: {annotation_folder} - Predicted Class: {classification}")

    overall_classifications = {class_name for class_name, count in annotation_class_counter.items() if count >= 2}
    print(f"Result for WSI {wsi_name}: {', '.join(overall_classifications)}, True: {ground_truth.get(wsi_name, 'Unknown')}")

    return overall_classifications


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'antrum_corpus_final.pth'
    class_names = ['antrum', 'corpus', 'intermediate']  # Include 'intermediate' in class_names
    num_classes = 2



    transform = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_model(model_path, num_classes)
    model.to(device)

    root = tk.Tk()
    root.withdraw()
    parent_dir = filedialog.askdirectory(title="Select the parent directory containing WSIs")

    if not parent_dir:
        print("No directory selected. Exiting.")
        return

    wsi_classes_filepath = 'wsi_classes.csv'
    wsi_classes_dict = {}
    csv_filepath = wsi_classes_filepath
    ground_truth = load_ground_truth(csv_filepath)
    with open(wsi_classes_filepath, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            wsi = row[0]
            classes = set(row[1].split(', '))
            wsi_classes_dict[wsi] = classes

    y_true = []
    y_pred = []

    for wsi_folder in os.listdir(parent_dir):
        wsi_path = os.path.join(parent_dir, wsi_folder)
        if os.path.isdir(wsi_path):
            overall_classifications = process_wsi(wsi_path, model, device, class_names, transform, ground_truth)
            true_classes = wsi_classes_dict.get(wsi_folder, set())
            y_true.append(true_classes)
            y_pred.append(overall_classifications)

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=class_names)

    # Convert y_true and y_pred to binary label vectors
    y_true_binary = mlb.fit_transform(y_true)
    y_pred_binary = mlb.transform(y_pred)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average=None)

    # Print the results
    for class_name, p, r, f in zip(class_names, precision, recall, f1):
        print(f"{class_name}: Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f:.4f}")


if __name__ == '__main__':
    main()

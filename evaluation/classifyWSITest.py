import os
import re
import csv
import torch
import timm
from torchvision import datasets, models
from torch import nn
import tkinter as tk
from PIL import Image
from collections import Counter, defaultdict
from torchvision import transforms
from tkinter import filedialog
from sklearn.metrics import accuracy_score


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
    model_path = '../antrum_corpus_final.pth'
    class_names = ['antrum', 'corpus']
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
    csv_file = "../image_classes.csv"
    ground_truth = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            wsi_folder, annotation_folder, class_name = row
            if wsi_folder not in ground_truth:
                ground_truth[wsi_folder] = {}
            ground_truth[wsi_folder][annotation_folder] = class_name

    # Include "intermediate" in the class names for accuracy tracking
    extended_class_names = class_names + ['intermediate']

    # Initialize total accuracies dictionary
    total_accuracies = defaultdict(list)

    # Initialize lists to store all true labels and predicted labels across all WSI folders
    all_y_true = []
    all_y_pred = []

    # Iterate over threshold values
    for threshold in [i / 100 for i in range(60, 96, 5)]:
        print(f"\nEvaluating for threshold: {threshold}")

        class_accuracies = defaultdict(list)

        # Iterate over each WSI folder
        for wsi_folder in os.listdir(base_dir):
            wsi_path = os.path.join(base_dir, wsi_folder)
            if os.path.isdir(wsi_path) and wsi_folder in ground_truth:
                #print(f"Analyzing WSI: {wsi_folder}")

                # Initialize predictions dictionary
                predictions = {}

                # Iterate over each 'annotation_x' folder and make predictions
                for annotation_folder in os.listdir(wsi_path):
                    annotation_path = os.path.join(wsi_path, annotation_folder)
                    if os.path.isdir(annotation_path) and re.match(r"annotation_\d+", annotation_folder):
                        image_votes = []
                        for image_file in os.listdir(annotation_path):
                            image_path = os.path.join(annotation_path, image_file)
                            predicted_class = predict_image(image_path, model, device, class_names, transform)
                            image_votes.append(predicted_class)

                        # Majority vote with threshold
                        vote_counts = Counter(image_votes)
                        total_votes = sum(vote_counts.values())
                        classification = "intermediate"
                        for class_name, count in vote_counts.items():
                            if count / total_votes >= threshold:
                                classification = class_name
                                break
                        predictions[annotation_folder] = classification

                # Append y_true and y_pred to the lists
                y_true = [ground_truth[wsi_folder][folder] for folder in ground_truth[wsi_folder].keys()]
                y_pred = [predictions[folder] for folder in ground_truth[wsi_folder].keys()]
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

                # Calculate accuracy per class including "intermediate"
                for class_name in extended_class_names:
                    class_true = [1 if label == class_name else 0 for label in y_true]
                    class_pred = [1 if label == class_name else 0 for label in y_pred]
                    class_accuracy = accuracy_score(class_true, class_pred)
                    class_accuracies[class_name].append(class_accuracy)

        # Calculate and print overall accuracy based on all individual predictions and ground truth labels
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        total_accuracies["overall"].append(overall_accuracy)
        print(f"Total Accuracy: {overall_accuracy * 100:.2f}%")

        for class_name in extended_class_names:
            class_accuracy = sum(class_accuracies[class_name]) / len(class_accuracies[class_name])
            total_accuracies[class_name].append(class_accuracy)
            print(f"Accuracy for class {class_name}: {class_accuracy * 100:.2f}%")

    # Print accuracies for each threshold including "intermediate"
    print("\nAccuracies for each threshold:")
    for i, threshold in enumerate([i / 100 for i in range(60, 96, 5)]):
        print(f"Threshold: {threshold}")
        print(f"Overall Accuracy: {total_accuracies['overall'][i] * 100:.2f}%")
        for class_name in extended_class_names:
            print(f"Accuracy for class {class_name}: {total_accuracies[class_name][i] * 100:.2f}%")


if __name__ == '__main__':
    main()

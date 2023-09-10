from collections import Counter, defaultdict

import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn
from PIL import Image
import os
import re

import tkinter as tk
from tkinter import filedialog



def load_model(model_path, num_classes):
    # Load the ResNet-18 model
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def predict_image(image_path, model, device, class_names, transform):
    # Open image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, 1)

    predicted_class = class_names[predicted_idx.item()]
    predicted_probability = probabilities[0][predicted_idx].item()

    return predicted_class, predicted_probability


def main():
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'resnet18_trained.pth'

    # Assuming you have the class names list from previous training
    class_names = ['antrum', 'corpus']  # Update this with your actual class names
    num_classes = len(class_names)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the trained model
    model = load_model(model_path, num_classes)
    model.to(device)

    # GUI for directory selection
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    main_dir = filedialog.askdirectory(title="Select the main directory containing annotation folders")

    # If the user cancels the directory selection, exit the program
    if not main_dir:
        print("No directory selected. Exiting.")
        return
    else:
        print(f"Analyzing WSI: {main_dir}")

    # Counter to record classifications for each annotation folder
    annotation_class_counter = Counter()



    # Iterate over each 'annotation_x' folder
    for annotation_folder in os.listdir(main_dir):
        annotation_path = os.path.join(main_dir, annotation_folder)

        # Ensure it's a directory and follows the pattern 'annotation_x'
        if os.path.isdir(annotation_path) and re.match(r"annotation_\d+", annotation_folder):

            # List to store classifications for this annotation folder
            image_votes = []

            for image_file in os.listdir(annotation_path):
                image_path = os.path.join(annotation_path, image_file)
                probable_class, _ = predict_image(image_path, model, device, class_names, transform)
                image_votes.append(probable_class)

            # Classify the annotation folder based on the threshold voting
            threshold = 0.6  # Set this to your preferred threshold
            vote_counts = Counter(image_votes)
            total_votes = sum(vote_counts.values())

            classification = None
            for class_name, count in vote_counts.items():
                if count / total_votes >= threshold:
                    classification = class_name
                    break

            if not classification:
                classification = "intermediate"

            # Increment the counter for the predicted class of the annotation folder
            annotation_class_counter[classification] += 1

            print(f"Annotation Folder: {annotation_folder} - Predicted Class: {classification}")

    # Determine the overall classifications for the main directory based on the counter
    overall_classifications = {class_name for class_name, count in annotation_class_counter.items() if count >= 3}

    # Print the overall classifications for the main directory
    print(f"\nCell Types found in WSI: {', '.join(overall_classifications)}")


if __name__ == '__main__':
    main()

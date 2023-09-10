import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from collections import Counter


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
    class_probabilities = {class_name: prob for class_name, prob in zip(class_names, probabilities[0].tolist())}

    # Check if probabilities exceed the threshold
    threshold = 0.6
    if class_probabilities[predicted_class] < threshold:
        predicted_class = "intermediate"

    return predicted_class, class_probabilities


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'resnet18_trained.pth'
    class_names = ['antrum', 'corpus']  # Update this with your actual class names

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_model(model_path, len(class_names))
    model.to(device)

    # GUI for file selection
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select images for classification")

    class_counter = Counter()

    for file_path in file_paths:
        predicted_class, class_probabilities = predict_image(file_path, model, device, class_names, transform)
        class_counter[predicted_class] += 1
        print(f"Image: {file_path}")
        print(f"Predicted Class: {predicted_class}")
        for class_name, probability in class_probabilities.items():
            print(f"Probability for {class_name}: {probability:.4f}")
        print('-' * 50)

    # Print the overall classification counts
    print("\nTotal occurrences for each class:")
    for class_name, count in class_counter.items():
        print(f"{class_name}: {count}")


if __name__ == '__main__':
    main()

import os
import tkinter as tk
from tkinter import filedialog
from collections import Counter
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

def load_model(model_path, num_classes):
    model = timm.create_model('legacy_xception', pretrained=False, num_classes=num_classes)
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
    subfolder_predictions = []

    for annotation_folder in os.listdir(wsi_folder_path):
        annotation_path = os.path.join(wsi_folder_path, annotation_folder)

        if os.path.isdir(annotation_path):
            class_counter = Counter()

            for image_file in os.listdir(annotation_path):
                image_path = os.path.join(annotation_path, image_file)
                predicted_class = predict_image(image_path, model, device, class_names, transform)
                class_counter[predicted_class] += 1

            most_common_class, _ = class_counter.most_common(1)[0]
            subfolder_predictions.append(most_common_class)
            print(f"Annotation Folder: {annotation_folder} - Predicted Class: {most_common_class}")

    overall_class_counter = Counter(subfolder_predictions)
    most_common_overall_class, _ = overall_class_counter.most_common(1)[0]

    return most_common_overall_class

def main():
    root = tk.Tk()
    root.withdraw()
    wsi_folder_path = filedialog.askdirectory(title="Select the WSI folder")

    if not wsi_folder_path:
        print("No folder selected. Exiting.")
        return

    model_path = 'inflamation_final_xception.pth'
    class_names = ['inflamed', 'noninflamed']
    num_classes = len(class_names)

    transform = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes)
    model.to(device)

    overall_classification = analyze_wsi_folder(wsi_folder_path, model, device, class_names, transform)
    print(f"\nOverall Predicted Class for WSI: {overall_classification}")

if __name__ == '__main__':
    main()

import os
import csv

# Define the base path
base_path = "/wsi-test"

# Initialize the CSV file
csv_file = "../image_classes_no_intermediate.csv"

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Get the list of first-level folders
    first_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Sort the first-level folders
    first_folders.sort()

    # Iterate through the first-level folders
    for first_folder in first_folders:
        first_folder_path = os.path.join(base_path, first_folder)

        # Walk through the first-level folder
        for root, dirs, files in os.walk(first_folder_path):
            # Check if the current folder is an annotation folder
            if "annotation_" in root:
                # Sort the files to ensure we are getting the first image
                files.sort()
                for file_name in files:
                    # Check if the file is an image
                    if file_name.endswith("antrum.png") or file_name.endswith("corpus.png"):
                        # Extract the class from the image name
                        class_name = file_name.split("_")[-1].replace(".png", "")
                        # Write the first folder name, annotation folder name, and class to the CSV file
                        writer.writerow([first_folder, os.path.basename(root), class_name])
                        # Break after writing the first image class in each folder
                        break

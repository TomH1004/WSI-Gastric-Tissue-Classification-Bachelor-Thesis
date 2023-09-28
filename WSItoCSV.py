import csv

# Dictionary to store unique classes for each WSI
wsi_classes = {}

# Read the CSV file
with open('image_classes_inflamation.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        wsi_name, _, class_name = row
        if wsi_name not in wsi_classes:
            wsi_classes[wsi_name] = set()
        wsi_classes[wsi_name].add(class_name)

# Write the results to a new CSV file
with open('wsi_classes_inflamation.csv', 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    for wsi_name, classes in wsi_classes.items():
        csv_writer.writerow([wsi_name, ', '.join(classes)])

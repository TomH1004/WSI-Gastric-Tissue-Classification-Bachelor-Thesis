import os

# Directory where the images are stored
base_dir = 'C:/Users/tomhe/Desktop/QuPath-Antrum-Corpus-Final/tiles'

# Initialize a dictionary to store the count of each class
class_counts = {}

# Initialize a counter for the total number of images
total_images = 0

# Iterate over each sub-directory
for dir_name in os.listdir(base_dir):
    if 'HE' in dir_name:
        dir_path = os.path.join(base_dir, dir_name)
        for sub_dir_name in os.listdir(dir_path):
            if 'annotation_' in sub_dir_name:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)

                # Iterate over each file in the sub-directory
                for file_name in os.listdir(sub_dir_path):
                    if file_name.endswith('.png'):
                        # Increment the total image counter
                        total_images += 1

                        # Extract the class name from the filename
                        class_name = file_name.split('_')[-1].replace('.png', '')

                        # Increment the counter for this class
                        if class_name not in class_counts:
                            class_counts[class_name] = 1
                        else:
                            class_counts[class_name] += 1

# Print out the total count for each class and all classes combined
for class_name, count in class_counts.items():
    print(f'Total for class {class_name}: {count}')

print(f'Total for all classes: {total_images}')

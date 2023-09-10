import os
from PIL import Image
import numpy as np

# Directory where the images are stored
base_dir = 'C:/Users/tomhe/Desktop/Dataset_Sorted/train'

# Initialize a counter
removed_images = 0

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
                        file_path = os.path.join(sub_dir_path, file_name)

                        # Open the image file
                        img = Image.open(file_path)

                        # Convert the image data to an array
                        data = np.array(img)

                        # Count the white (also shades of whites)
                        # pixels (Here, [200,200,200] is the threshold for white, can change as per need)
                        white_pixels = np.sum(data > [200, 200, 200])
                        total_pixels = np.prod(data.shape)

                        # If more than 90% pixels are white, delete the image
                        if white_pixels / total_pixels > 0.9:
                            os.remove(file_path)
                            removed_images += 1
                            print(f'Removed {file_path}')

print(f'Removed total {removed_images} images.')
import os
import shutil

# The root directory where the .png files are currently stored
root_dir = 'C:/Users/tomhe/Desktop/Exported_Images_Inflamation'

# The directory where the files should be copied to
dst_dir = '../data/inflammation/dataset_inflammation'

# The names of the classes (and also the subfolders)
classes = ['inflamed', 'noninflamed']

# Initialize counters for each class
counters = {cls: 0 for cls in classes}

# Make sure the subfolders exist
for cls in classes:
    os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)

# Iterate over each HE sub-directory in root directory
for dir_name in os.listdir(root_dir):
    if 'HE' in dir_name:
        dir_path = os.path.join(root_dir, dir_name)
        for sub_dir_name in os.listdir(dir_path):
            if 'annotation_' in sub_dir_name:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)

                # Iterate over all files in the annotation sub-directory
                for file_name in os.listdir(sub_dir_path):
                    # If the file is a .png file
                    if file_name.endswith('.png'):
                        # Determine the class of the image based on its name
                        for cls in classes:
                            if f"_{cls}.png" in file_name:
                                # If the class matches, copy the file into the corresponding subfolder
                                shutil.copy(os.path.join(sub_dir_path, file_name), os.path.join(dst_dir, cls, file_name))

                                # Increment the counter for this class
                                counters[cls] += 1

# Print the counters
for cls, count in counters.items():
    print(f"Copied {count} images of class '{cls}'")

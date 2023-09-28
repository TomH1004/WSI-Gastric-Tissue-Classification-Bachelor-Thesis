import os
import shutil

# Define the path where you want to delete folders
path = "C:\\Users\\tomhe\\Desktop\\QuPath-Antrum-Corpus-Final\\tiles"

# List all folders in the specified path
folders = os.listdir(path)

# Loop through each folder
for folder in folders:
    folder_path = os.path.join(path, folder)

    # Check if the folder name ends with "HE" or "HE!" and the folder is empty
    if (folder.endswith("HE") or folder.endswith("HE!")) and os.path.isdir(folder_path) and not os.listdir(folder_path):
        # Delete the empty folder
        shutil.rmtree(folder_path)
        print(f"Deleted empty folder: {folder_path}")

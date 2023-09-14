import os
import shutil
import pandas as pd

# Directories
source_directory = 'C:/Users/tomhe/Desktop/Dataset_AntrumCorpus_With_InflamedSlides'
target_directory = 'C:/Users/tomhe/Desktop/Dataset_Sorted_with_Inflamed'
csv_path = 'C:/Users/tomhe/Desktop/data_split.csv'  # Change this to the path of your CSV file


# Extracting folder names for each category
def extract_folders(row):
    category, folders_str = row.split(":")
    folders = folders_str.replace("[", "").replace("]", "").replace("'", "").replace("\n", "").split()
    return category.strip(), folders


# Read the CSV and extract folder names
data_split_df = pd.read_csv(csv_path)
categories = {}
for index, row in data_split_df.iterrows():
    category, folders = extract_folders(row[0])
    categories[category] = folders

# Manually extracting the 'train' folders from the CSV content
train_folders_str = data_split_df.columns[0].split(":")[1]
train_folders = train_folders_str.replace("[", "").replace("]", "").replace("'", "").replace("\n", "").split()
categories['train'] = train_folders

# Create target subdirectories if they don't exist
for category in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_directory, category), exist_ok=True)

# Move folders to respective subdirectories
for category, folders in categories.items():
    for folder in folders:
        source_folder_path = os.path.join(source_directory, folder)
        target_folder_path = os.path.join(target_directory, category, folder)

        if os.path.exists(source_folder_path):
            shutil.move(source_folder_path, target_folder_path)
        else:
            print(f"Folder {folder} does not exist in source directory.")

print("Folders sorted successfully!")

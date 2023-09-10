This guide outlines the steps to create and prepare a dataset for training.

## Step 1: Remove Empty Images

Run `removeEmptyImages.py` to remove images with predominantly white pixels

## Step 2: Sort Images

Run `sortImages.py` to automatically organize exported images into project directories

## Step 3: Data Augmentation

Run `dataAugmentation.py` to generate three augmented images per original image with 90-degree rotations

## Clearing Project Dataset

Run `clearDataset.py` to remove every image from the dataset

Run `cleareAugmentedImages.py` to remove every augmented image from the dataset

## Count Images

Run `countImages.py` to get a count of images per directory/class

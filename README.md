# Gastric and Inflammatory Classification Project

## Overview
This repository contains the codebase for a project focused on the classification of gastric regions and inflammation. Due to size limitations on GitHub, the datasets and the entire project are hosted on the university chair's file server.

## Project Structure
The project is structured as follows:
- 📁 **Project Root Directory**
    - 📁 **data**
        - 📁 **dataset**
        - 📁 **dataset_validation**
        - 📁 **dataset_test**
        - 📁 **dataset_inflamation**
        - 📁 **dataset_inflamation_validation**
        - 📁 **dataset_inflamation_test**
        - 📁 **wsi_test**
        - 📁 **wsi_inflamation_test**
    - 📁 **csv**
        - 📄 (multiple CSV files)
    - 📁 **evaluation**
        - 📄 (multiple scripts for testing purposes)
    - 📁 **models**
        - 📄 (all trained models)
    - 📄 **classifyFiles.py**
    - 📄 **classifyWSI.py**
    - 📄 **classifyInflamedWSI.py**
    - 📄 **trainValidateModel.py**
 
### Main Directory Files
- `classifyFiles.py`: Classifies selected tiles.
- `classifyWSI.py`: Classifies selected Whole Slide Images (WSIs) for gastric region.
- `classifyInflamedWSI.py`: Classifies selected WSIs for inflammation.
- `trainValidateModel.py`: Trains and validates the models.

## Accessing the Project
The datasets and the entire project are located at the university chair's file server. Please contact the repository owner or the university chair for access to the server and further instructions on how to access the datasets and the project files.

## Evaluation Scripts
The `evaluation` directory contains multiple scripts that are used for testing purposes. These scripts help in assessing the performance and accuracy of the trained models on the test datasets.
- `testAnnotation.py`: Tests performance on particle level for gastric region.
- `testAnnotationInflammation.py`: Does the same for inflammation classification.
- `testModel.py`: Evaluates the performance of the model on the tiles from the test set.
- `testWSI.py`: Evaluates performance on WSI-level for test set on gastric region classification.
- `testWSI_inflammation.py`: Does the same but for inflammation classifications.

## Models
The `models` directory houses all the trained models used for gastric region classification and inflammatory classification. These models have been trained and validated using the datasets provided in the respective dataset directories.

## Configuration
For many scripts within this project, the paths to directories need to be adapted to use certain architectures or datasets. Additionally, it might be necessary to uncomment and comment out code for different architectures. Please ensure to review and modify the script paths and configurations as needed to suit the specific use case and environment.

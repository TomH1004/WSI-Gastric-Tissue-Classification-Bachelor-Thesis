import os


def delete_augmented_images(folder_path, folder_name):
    for image_filename in os.listdir(folder_path):
        if any(image_filename.lower().endswith(f'_augmented_rotation_{angle}_{folder_name}.png') for angle in
               [90, 180, 270]):
            image_path = os.path.join(folder_path, image_filename)
            os.remove(image_path)
            print(f"Deleted augmented image: {image_filename}")


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(main_dir, "../data/dataset_validation")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        for folder_name in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder_name)
            if os.path.isdir(folder_path):
                delete_augmented_images(folder_path, folder_name)
    else:
        print("Dataset directory not found.")

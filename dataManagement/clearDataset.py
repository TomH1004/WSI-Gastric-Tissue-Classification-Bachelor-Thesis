import os


def clear_image_folder(folder_path):
    for image_filename in os.listdir(folder_path):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_filename)
            os.remove(image_path)
            print(f"Deleted image: {image_filename}")


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(main_dir, "../dataset")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        for folder_name in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder_name)
            if os.path.isdir(folder_path):
                clear_image_folder(folder_path)
    else:
        print("Dataset directory not found.")

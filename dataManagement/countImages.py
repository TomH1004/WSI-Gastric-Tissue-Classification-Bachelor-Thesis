import os


def count_images_per_folder(root_dir):
    image_count = {}

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            image_count[folder_name] = len(
                [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

    return image_count


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(main_dir, "../data/dataset_inflamation")
    validation_dir = os.path.join(main_dir, "../data/dataset_inflamation_validation")
    test_dir = os.path.join(main_dir, "../data/dataset_inflamation_test")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        image_counts_dataset = count_images_per_folder(dataset_dir)
        total_images_dataset = sum(image_counts_dataset.values())

        print("Image count for train directory:")
        for folder_name, count in image_counts_dataset.items():
            print(f"Folder: {folder_name}, Image Count: {count}")
        print(f"Total Images: {total_images_dataset}\n")

        image_counts_validation = count_images_per_folder(validation_dir)
        total_images_validation = sum(image_counts_validation.values())

        print("Image count for validation directory:")
        for folder_name, count in image_counts_validation.items():
            print(f"Folder: {folder_name}, Image Count: {count}")
        print(f"Total Images: {total_images_validation}\n")

        image_counts_test = count_images_per_folder(test_dir)
        total_images_test = sum(image_counts_test.values())

        print("Image count for test directory:")
        for folder_name, count in image_counts_test.items():
            print(f"Folder: {folder_name}, Image Count: {count}")
        print(f"Total Images: {total_images_test}\n")
    else:
        print("Dataset directory not found.")

import os
import shutil
import random


def move_images_to_validation(src_dir, dest_dir, percentage):
    for class_folder in os.listdir(src_dir):
        class_src_dir = os.path.join(src_dir, class_folder)
        class_dest_dir = os.path.join(dest_dir, class_folder)

        if not os.path.exists(class_dest_dir):
            os.makedirs(class_dest_dir)

        images = [img for img in os.listdir(class_src_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images_to_move = int(percentage * len(images))

        images_to_move = random.sample(images, num_images_to_move)
        for img in images_to_move:
            src_path = os.path.join(class_src_dir, img)
            dest_path = os.path.join(class_dest_dir, img)
            shutil.move(src_path, dest_path)
            print(f"Moved {img} to {class_dest_dir}")


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(main_dir, "../data/inflammation/dataset_inflammation")
    validation_dir = os.path.join(main_dir, "../data/inflammation/dataset_inflammation_validation")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)

        move_images_to_validation(dataset_dir, validation_dir, 0.15)
    else:
        print("Dataset directory not found.")

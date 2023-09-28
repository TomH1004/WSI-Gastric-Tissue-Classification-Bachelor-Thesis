import os
from PIL import Image


def augment_and_save_images(folder_path, folder_name):
    for image_filename in os.listdir(folder_path):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_filename)
            image = Image.open(image_path)

            # Apply rotations of 90, 180, and 270 degrees
            rotated_images = []
            for angle in [90, 180, 270]:
                rotated_image = image.rotate(angle)
                rotated_images.append(rotated_image)

            # Generate new names for the augmented images
            for angle, rotated_image in zip([90, 180, 270], rotated_images):
                new_image_filename = image_filename.replace(f'_{folder_name}.png',
                                                            f'_augmented_rotation_{angle}_{folder_name}.png')
                new_image_path = os.path.join(folder_path, new_image_filename)

                # Save the rotated image
                rotated_image.save(new_image_path)
                print(f"Saved rotated image: {new_image_filename}")


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(main_dir, "../2HE")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        for folder_name in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder_name)
            if os.path.isdir(folder_path):
                augment_and_save_images(folder_path, folder_name)
    else:
        print("Dataset directory not found.")

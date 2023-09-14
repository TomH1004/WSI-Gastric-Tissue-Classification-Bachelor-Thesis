import os
from PIL import Image
from torchvision import transforms
import torch


def create_pth_dataset(root_dirs, batch_size=1000):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    output_dir = 'dataset_compressed'
    os.makedirs(output_dir, exist_ok=True)

    for root_dir in root_dirs:
        data = []
        labels = []
        class_names = sorted(os.listdir(root_dir))
        batch_num = 0

        root_output_dir = os.path.join(output_dir, os.path.basename(root_dir))
        os.makedirs(root_output_dir, exist_ok=True)

        print(f"Processing dataset in {root_dir}...")

        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            print(f"  Processing class {class_name}...")
            for image_name in os.listdir(class_dir):
                if image_name.endswith('.png'):
                    image_path = os.path.join(class_dir, image_name)
                    image = Image.open(image_path)
                    image = transform(image)
                    data.append(image)
                    labels.append(label)

                    # Save the batch when it reaches the batch size or at the end of the dataset
                    if len(data) >= batch_size or (
                            label == len(class_names) - 1 and image_name == os.listdir(class_dir)[-1]):
                        dataset = {'data': torch.stack(data), 'labels': torch.tensor(labels)}
                        torch.save(dataset, f'{root_output_dir}/batch_{batch_num}.pth')
                        print(f'Saved batch_{batch_num}.pth in {root_output_dir}')

                        # Reset data and labels for the next batch
                        data = []
                        labels = []
                        batch_num += 1


root_dirs = [
    'dataset',
    'dataset_inflamation',
    'dataset_test',
    'dataset_inflamation_test',
    'dataset_validation',
    'dataset_inflamation_validation'
]

create_pth_dataset(root_dirs, batch_size=1000)

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split


def create_balanced_sampler(dataset):
    targets = []
    for line in dataset.lines:
        spoof_type = int(line.strip().split()[1])
        label = 0 if spoof_type == 0 else 1
        targets.append(label)

    targets = torch.tensor(targets)
    class_counts = torch.bincount(targets)
    real = class_counts[0].item()
    fake = class_counts[1].item()

    weight_real = 1. / real
    weight_fake = 1. / fake

    sample_weights = torch.zeros(len(targets))
    sample_weights[targets == 0] = weight_real
    sample_weights[targets == 1] = weight_fake

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

class CelebASpoofDataset(Dataset):
    def __init__(self, root_dir, data_lines, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.lines = data_lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        items = line.split()
        img_rel_path = items[0]
        spoof_type = int(items[1])

        image_path = os.path.join(self.root_dir, img_rel_path)

        image = cv2.imread(image_path)
        if image is None:
            return self.__getitem__((idx+1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256))

        if spoof_type == 0:
            depth_map =np.ones((32,32), dtype=np.float32)
            binary_label = 1.0
        else:
            depth_map = np.zeros((32,32), dtype=np.float32)
            binary_label = 0.0

        if self.transform:
            augmented = self.transform(image = image)
            image = augmented['image']

        depth_map = torch.from_numpy(depth_map).unsqueeze(0)
        return image, depth_map, torch.tensor(binary_label, dtype=torch.float32)



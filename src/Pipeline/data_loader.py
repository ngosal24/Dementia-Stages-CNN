import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from collections import Counter
import numpy as np
import torch

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def get_labels_from_subset(subset):
    return [subset.dataset.targets[i] for i in subset.indices]

def get_dataloaders_with_validation(data_dir, batch_size=32, val_split=0.2):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    full_train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    val_size = int(val_split * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Important: override transform for validation set
    val_dataset.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_train_dataset.classes, train_dataset

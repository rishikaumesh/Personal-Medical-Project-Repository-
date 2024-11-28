import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

def load_data(train_dir, val_dir, test_dir, img_size=224, batch_size=32, use_weighted_sampler=True):
    """
    Load train, validation, and test datasets and return their DataLoaders.
    """
    # Debugging statements
    print(f"train_dir type: {type(train_dir)}, val_dir type: {type(val_dir)}, test_dir type: {type(test_dir)}")
    print(f"train_dir: {train_dir}, val_dir: {val_dir}, test_dir: {test_dir}")

    if not isinstance(train_dir, str) or not isinstance(val_dir, str) or not isinstance(test_dir, str):
        raise ValueError("train_dir, val_dir, and test_dir must be strings representing paths.")

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    if use_weighted_sampler:
        class_counts = [train_dataset.targets.count(0), train_dataset.targets.count(1)]
        class_weights = [1.0 / count for count in class_counts]
        sample_weights = [class_weights[label] for label in train_dataset.targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of images in training set: {len(train_dataset)}")
    print(f"Number of images in validation set: {len(val_dataset)}")
    print(f"Number of images in test set: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

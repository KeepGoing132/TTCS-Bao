"""
RAF-DB Dataset Module
Handles loading and preprocessing of RAF-DB (Real-world Affective Faces Database)
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import config


def get_raf_db_transforms(train=False):
    """
    Get RAF-DB optimized transforms.
    
    RAF-DB uses:
    - Image size: 224x224
    - Normalization: Custom statistics from RAF-DB
    - Data augmentation for training
    """
    transform_list = []
    
    if train:
        # Training augmentations
        transform_list.extend([
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05)
            ),
        ])
    else:
        # Validation/Test transforms (minimal)
        transform_list.append(
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        )
    
    # Normalize with RAF-DB statistics
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.RAF_DB_MEAN,
            std=config.RAF_DB_STD
        )
    ])
    
    return transforms.Compose(transform_list)


def get_dataloaders(batch_size=None):
    """
    Create train, validation, and test dataloaders for RAF-DB.
    
    Args:
        batch_size: Batch size for dataloaders (default: config.BATCH_SIZE)
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_transform = get_raf_db_transforms(train=True)
    val_transform = get_raf_db_transforms(train=False)
    test_transform = get_raf_db_transforms(train=False)
    
    # Load full training dataset
    full_dataset = datasets.ImageFolder(root=config.TRAIN_DIR)
    total_samples = len(full_dataset)
    val_size = int(total_samples * config.VAL_SPLIT)
    train_size = total_samples - val_size
    
    # Split dataset deterministically
    torch.manual_seed(config.SEED)
    all_indices = torch.randperm(total_samples).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
    # Create datasets
    train_dataset = Subset(
        datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform),
        train_indices
    )
    val_dataset = Subset(
        datasets.ImageFolder(root=config.TRAIN_DIR, transform=val_transform),
        val_indices
    )
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=test_transform
    )
    
    # Create dataloaders with flexible batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print("Dataset loaded successfully!")
    
    # Test first batch
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Label sample: {labels[:5]}")
        break
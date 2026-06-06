
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import config


def get_raf_db_transforms(train=False):
    transform_list = []
    
    if train:
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
        transform_list.append(
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        )
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.RAF_DB_MEAN,
            std=config.RAF_DB_STD
        )
    ])
    
    return transforms.Compose(transform_list)


def get_dataloaders(batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_transform = get_raf_db_transforms(train=True)
    val_transform = get_raf_db_transforms(train=False)
    test_transform = get_raf_db_transforms(train=False)
    
    full_dataset = datasets.ImageFolder(root=config.TRAIN_DIR)
    total_samples = len(full_dataset)
    val_size = int(total_samples * config.VAL_SPLIT)
    train_size = total_samples - val_size
    
    torch.manual_seed(config.SEED)
    all_indices = torch.randperm(total_samples).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
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
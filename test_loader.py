# test_loader.py
"""
This script loads and visualizes samples from the TuSimple lane detection dataset.

Modules:
    - matplotlib.pyplot: For displaying images and masks.
    - torch.utils.data.DataLoader: For batching and shuffling dataset samples.
    - dataset_loader.TuSimpleDataset: Custom dataset class for TuSimple lane detection data.
    - torchvision.transforms: For preprocessing and transforming images.

Workflow:
    1. Defines a transformation pipeline to resize and convert images to tensors.
    2. Loads the TuSimple dataset with the specified transform and creates a DataLoader.
    3. Iterates over one batch from the DataLoader, displaying the input image and its corresponding lane mask side by side using matplotlib.

Usage:
    Run this script to visualize a sample image and its lane mask from the TuSimple dataset.
"""
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import TuSimpleDataset
import torchvision.transforms as T

# Path to "tusimple/TUSimple"
dataset_path = 'tusimple/TUSimple'

# Define transform
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 512)),
    T.ToTensor()
])

# Create dataset and dataloader
dataset = TuSimpleDataset(dataset_path, split='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Show one batch
for img, mask in dataloader:
    img_np = img[0].permute(1, 2, 0).numpy()   # C,H,W -> H,W,C
    mask_np = mask[0].numpy()

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img_np)

    plt.subplot(1, 2, 2)
    plt.title("Lane Mask")
    plt.imshow(mask_np, cmap='gray')

    plt.show()
    break

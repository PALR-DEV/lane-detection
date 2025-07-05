


"""
TuSimple Dataset Loader for Lane Detection

This module implements a PyTorch Dataset class for loading and preprocessing 
the TuSimple lane detection dataset. The TuSimple dataset is a benchmark dataset
for lane detection tasks in autonomous driving.

Dataset Structure:
- Images: RGB road scenes with lane markings
- Annotations: JSON files containing lane coordinates as polynomials
- Format: Each lane is represented as x-coordinates at fixed y-positions (h_samples)

Key Features:
- Supports both training and testing splits
- Converts polynomial lane representations to binary masks
- Applies data augmentation and preprocessing transforms
- Handles multiple annotation files per split

Author: Lane Detection Project
Date: Dataset implementation for TuSimple lane detection
"""

# Import required libraries
from torch.utils.data import Dataset, DataLoader  # PyTorch dataset utilities
import torchvision.transforms as T                # Image preprocessing transforms
import os, glob, json, cv2, torch                # File I/O, computer vision, tensor operations

class TuSimpleDataset(Dataset):
    """
    PyTorch Dataset for TuSimple Lane Detection Dataset
    
    The TuSimple dataset contains road images with lane annotations stored as JSON files.
    Each annotation contains:
    - lanes: List of lane coordinates (x-values at specific y-positions)
    - h_samples: Fixed y-coordinates where x-values are measured
    - raw_file: Relative path to the corresponding image file
    
    This class:
    1. Loads image and annotation pairs
    2. Converts lane coordinates to binary segmentation masks
    3. Applies preprocessing transforms
    4. Returns (image, mask) pairs for training
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize the TuSimple dataset loader
        
        Args:
            root_dir (str): Path to the TuSimple dataset root directory
                           Expected structure: root_dir/
                           ├── train_set/
                           │   ├── label_data_*.json
                           │   └── clips/
                           └── test_set/
                               ├── test_label.json
                               └── clips/
            split (str): Dataset split to load ('train' or 'test')
            transform (callable, optional): Transform to apply to images and masks
        """
        self.transform = transform
        self.samples = []  # List to store all annotation entries
        
        # =================================================================
        # ANNOTATION FILE DISCOVERY
        # =================================================================
        
        if split == 'train':
            # Training set: Multiple JSON files with different date stamps
            label_folder = os.path.join(root_dir, 'train_set')
            # Find all label files matching pattern: label_data_*.json
            # Examples: label_data_0313.json, label_data_0531.json, etc.
            json_files = glob.glob(os.path.join(label_folder, 'label_data_*.json'))
        else:
            # Test set: Specific named JSON files
            label_folder = os.path.join(root_dir, 'test_set')
            json_files = [
                os.path.join(label_folder, 'test_label.json'),      # Original test labels
                os.path.join(label_folder, 'test_label_new.json')   # Updated test labels
            ]
        
        # =================================================================
        # ANNOTATION LOADING
        # =================================================================
        
        # Load annotations from all JSON files
        # Each file contains multiple lines, each line is a separate JSON object
        for jf in json_files:
            if os.path.exists(jf):  # Check if file exists before opening
                with open(jf, 'r') as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            # Each line is a JSON object containing one image's annotation
                            # Format: {"lanes": [...], "h_samples": [...], "raw_file": "..."}
                            self.samples.append(json.loads(line))
        
        # Base directory for loading images
        # Images are stored in: root_dir/train_set/clips/ or root_dir/test_set/clips/
        self.img_base = os.path.join(root_dir, split + '_set')
        
        print(f"Loaded {len(self.samples)} samples from {split} split")


    def __len__(self):
        """
        Return the total number of samples in the dataset
        
        Returns:
            int: Number of annotated images in the dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample (image, mask) pair from the dataset
        
        This method:
        1. Loads the image from file
        2. Extracts lane annotations
        3. Creates a binary segmentation mask from lane coordinates
        4. Applies preprocessing transforms
        5. Returns the processed image-mask pair
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, mask) where:
                - image: RGB image tensor of shape (3, H, W) after transforms
                - mask: Binary mask tensor of shape (H, W) with lane pixels = 1
        """
        
        # =================================================================
        # ANNOTATION EXTRACTION
        # =================================================================
        
        # Get annotation entry for this index
        entry = self.samples[idx]
        
        # Extract lane information from annotation
        lanes     = entry['lanes']        # List of lanes, each lane is list of x-coordinates
        h_samples = entry['h_samples']    # Fixed y-coordinates (height samples)
        raw_file  = entry['raw_file']     # Relative path to image file
        
        # Example annotation structure:
        # {
        #   "lanes": [
        #     [1080, 1040, 1000, ...],  # Lane 1: x-coords at each h_sample
        #     [1500, 1460, 1420, ...],  # Lane 2: x-coords at each h_sample
        #     [-2, -2, -2, ...]         # Lane 3: -2 means no lane at this y-position
        #   ],
        #   "h_samples": [160, 170, 180, ..., 710],  # y-coordinates
        #   "raw_file": "clips/0530/1492626047222176976_0/20.jpg"
        # }
        
        # =================================================================
        # IMAGE LOADING
        # =================================================================
        
        # Construct full path to image file
        img_path = os.path.join(self.img_base, raw_file)
        
        # Load image using OpenCV and convert BGR to RGB
        # OpenCV loads images in BGR format, but PyTorch expects RGB
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        img = img[:, :, ::-1]  # Convert BGR to RGB by reversing channel order
        H, W, _ = img.shape    # Get original image dimensions
        
        # =================================================================
        # MASK GENERATION
        # =================================================================
        
        # Create empty binary mask with same spatial dimensions as image
        # mask[y, x] = 1 if there's a lane pixel at position (x, y), 0 otherwise
        mask = torch.zeros((H, W), dtype=torch.float32)
        
        # Convert lane coordinates to binary mask
        for lane in lanes:
            # Iterate through each lane's x-coordinates
            for x, y in zip(lane, h_samples):
                # Skip invalid coordinates (marked as -2 in TuSimple)
                if x >= 0:
                    # Draw a small circle at lane position to create thick lane lines
                    # This makes lanes more visible and easier for the model to learn
                    # radius=2: Creates 5x5 pixel circles for better visibility
                    # color=(1,): White pixels (value=1) in binary mask
                    # thickness=-1: Fill the circle completely
                    cv2.circle(
                        mask.numpy(), 
                        (int(x), int(y)), 
                        radius=2, 
                        color=(1,), 
                        thickness=-1
                    )
        
        # =================================================================
        # PREPROCESSING TRANSFORMS
        # =================================================================
        
        # Apply transforms if provided (resize, normalize, etc.)
        if self.transform:
            # Transform the image (straightforward)
            img = self.transform(img)
            
            # Transform the mask (more complex due to single channel)
            # 1. Add channel dimension: H×W → H×W×1 (required by transforms)
            m = mask.unsqueeze(2).numpy()
            
            # 2. Apply transform (converts to tensor and resizes)
            # Multiply by 255 to convert [0,1] to [0,255] range temporarily
            # Convert to long then back to float for numerical stability
            m = (self.transform(m) * 255).long()[0]  # Remove channel dim after transform
            
            # 3. Normalize back to [0,1] range
            mask = m.float() / 255.0
        
        return img, mask

# =============================================================================
# USAGE EXAMPLE AND TESTING
# =============================================================================

if __name__ == '__main__':
    """
    Example usage of the TuSimpleDataset class
    
    This section demonstrates how to:
    1. Create a dataset instance with transforms
    2. Set up a DataLoader for batch processing
    3. Load and inspect a batch of data
    
    Run this file directly to test the dataset implementation:
    python dataset_loader.py
    """
    
    # Define preprocessing transforms
    # These match the transforms used in training
    tf = T.Compose([
        T.ToPILImage(),         # Convert numpy array to PIL Image
        T.Resize((256, 512)),   # Resize to network input size
        T.ToTensor(),           # Convert to tensor and normalize to [0,1]
    ])
    
    # Create dataset instance
    # Note: Update the path to match your actual TuSimple dataset location
    ds = TuSimpleDataset('/mnt/data/tusimple/TUSimple', split='train', transform=tf)
    
    # Create DataLoader for batch processing
    dl = DataLoader(
        ds, 
        batch_size=4,      # Process 4 images at once
        shuffle=True,      # Randomize order
        num_workers=4      # Use 4 processes for parallel loading
    )
    
    # Load one batch and print shapes
    imgs, masks = next(iter(dl))
    print(f"Image batch shape: {imgs.shape}")    # Expected: [4, 3, 256, 512]
    print(f"Mask batch shape: {masks.shape}")    # Expected: [4, 256, 512]
    
    # Print some statistics
    print(f"Image value range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"Lane pixels per mask: {masks.sum(dim=(1,2))}")  # Count of lane pixels per image

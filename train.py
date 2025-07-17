"""
Lane Detection Training Script

This script trains a U-Net model for lane detection using the TuSimple dataset.
The training process includes:
1. Data loading and preprocessing
2. Model initialization
3. Loss function and optimizer setup
4. Training loop with backpropagation
5. Model checkpointing

Author: Lane Detection Project
Date: Training implementation for autonomous vehicle lane detection
"""

# Import custom modules
from dataset_loader import TuSimpleDataset  # Custom dataset class for TuSimple data
from model import SimpleUNet               # Our U-Net model architecture

# Import PyTorch components
from torch.utils.data import DataLoader    # Efficient batch loading and shuffling
import torchvision.transforms as T         # Image preprocessing transformations
import torch                              # Core PyTorch library
import torch.nn as nn                     # Neural network modules and loss functions
import torch.optim as optim              # Optimization algorithms

# Entry point for safe multiprocessing on macOS/Windows
if __name__ == '__main__':

    # =============================================================================
    # DATA PREPROCESSING PIPELINE
    # =============================================================================

    # Transform pipeline: Prepares raw images for neural network training
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 512)),
        T.ToTensor()
    ])

    # =============================================================================
    # DATASET AND DATALOADER SETUP
    # =============================================================================

    dataset = TuSimpleDataset('tusimple/TUSimple', split='train', transform=transform)
    print(f"Loaded {len(dataset)} samples from train split")

    train_loader = DataLoader(
        dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=2  # Set to 0 if you still get multiprocessing issues
    )

    # =============================================================================
    # MODEL, LOSS FUNCTION, AND OPTIMIZER SETUP
    # =============================================================================

    device = torch.device("mps" if torch.backendsmps.is_available() else "cpu")

    model = SimpleUNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # =============================================================================
    # TRAINING LOOP
    # =============================================================================

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            preds = model(imgs).squeeze(1)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs} - Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        checkpoint_path = f'model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    print("Training completed!")
    print("All model checkpoints saved. Use the latest or best-performing model for inference.")

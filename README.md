# 🛣️ Lane Detection with U-Net

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TuSimple](https://img.shields.io/badge/Dataset-TuSimple-orange.svg)](https://github.com/TuSimple/tusimple-benchmark)

A deep learning implementation for autonomous vehicle lane detection using U-Net architecture and the TuSimple benchmark dataset. This project demonstrates semantic segmentation for real-time lane marking detection in highway driving scenarios.

## 📋 Table of Contents

- [🔍 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [📊 Dataset](#-dataset)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🧠 Model Details](#-model-details)
- [📚 Code Walkthrough](#-code-walkthrough)
- [🔧 Training Process](#-training-process)
- [📈 Performance](#-performance)
- [🛠️ Customization](#️-customization)
- [📖 References](#-references)

## 🔍 Overview

Lane detection is a critical component of Advanced Driver Assistance Systems (ADAS) and autonomous vehicles. This project implements a simplified U-Net architecture to perform pixel-level binary segmentation, identifying lane markings in road images.

### Key Features

- **🎯 Semantic Segmentation**: Pixel-level lane classification using U-Net
- **🏃 Real-time Ready**: Lightweight architecture optimized for inference speed
- **📊 Benchmark Dataset**: Trained and tested on TuSimple dataset
- **🔄 End-to-End Pipeline**: Complete data loading, training, and visualization
- **📱 Modular Design**: Clean, extensible codebase with comprehensive documentation

### Technical Highlights

- **Architecture**: Simplified U-Net with skip connections
- **Loss Function**: Binary Cross-Entropy with Logits Loss
- **Optimizer**: Adam with learning rate 1e-4
- **Input Resolution**: 256×512 pixels (optimized for speed/accuracy balance)
- **Output**: Binary segmentation mask (lane vs. background)

## 🏗️ Architecture

### U-Net Overview

Our simplified U-Net consists of two main components:

#### 🔽 Encoder (Contracting Path)
- **Purpose**: Capture context and extract hierarchical features
- **Operations**: Convolution + pooling for downsampling
- **Feature Evolution**: 3 → 32 → 64 channels

#### 🔼 Decoder (Expanding Path)  
- **Purpose**: Precise localization and upsampling
- **Operations**: Transposed convolution + concatenation
- **Feature Evolution**: 64 → 32 → 1 channel

#### 🔗 Skip Connections
- **Purpose**: Preserve fine-grained spatial information
- **Implementation**: Concatenate encoder features with decoder features
- **Benefit**: Combines global context with local details

```
Input (3, 256, 512)
        ↓
    DoubleConv (32)
        ↓ ←――――――――――――┐
    MaxPool2d          │ Skip Connection
        ↓              │
    DoubleConv (64)    │
        ↓              │
    ConvTranspose2d    │
        ↓              │
    Concatenate ←――――――┘
        ↓
    DoubleConv (32)
        ↓
    Conv2d (1)
        ↓
Output (1, 256, 512)
```

## 📊 Dataset

### TuSimple Lane Detection Dataset

The TuSimple dataset is a benchmark for lane detection research, featuring:

- **🚗 Real Highway Data**: Collected from actual highway driving
- **📏 Resolution**: Original 720×1280 pixels (resized to 256×512 for training)
- **🏷️ Annotation Format**: JSON files with polynomial lane representations
- **📂 Structure**: Separate training and testing splits

#### Dataset Statistics
- **Training Set**: ~3,600 images
- **Test Set**: ~2,800 images  
- **Lane Types**: 2-4 lanes per image
- **Conditions**: Various lighting and weather conditions

#### Annotation Format
```json
{
  "lanes": [
    [1080, 1040, 1000, ...],  // Lane 1 x-coordinates
    [1500, 1460, 1420, ...],  // Lane 2 x-coordinates
    [-2, -2, -2, ...]         // Lane 3 (invisible)
  ],
  "h_samples": [160, 170, 180, ..., 710],  // y-coordinates
  "raw_file": "clips/0530/1492626047222176976_0/20.jpg"
}
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv lane_detection_env
source lane_detection_env/bin/activate  # On Windows: lane_detection_env\Scripts\activate

# Install dependencies
pip install torch torchvision opencv-python matplotlib numpy
```

### Dataset Setup

1. **Download TuSimple Dataset**:
   ```bash
   # Download from: https://github.com/TuSimple/tusimple-benchmark
   # Extract to: tusimple/TUSimple/
   ```

2. **Verify Structure**:
   ```
   tusimple/TUSimple/
   ├── train_set/
   │   ├── label_data_0313.json
   │   ├── label_data_0531.json
   │   ├── label_data_0601.json
   │   └── clips/
   └── test_set/
       ├── test_label.json
       └── clips/
   ```

### Training

```bash
# Start training
python train.py

# Monitor progress
# Training progress will be displayed in terminal
# Model checkpoints saved as model_epoch_*.pth
```

### Testing/Visualization

```bash
# Visualize dataset samples
python test_loader.py

# This will display:
# - Original road image
# - Generated binary lane mask
```

## 📁 Project Structure

```
lane_detection/
├── 📄 README.md              # This comprehensive guide
├── 🧠 model.py               # U-Net architecture implementation
├── 📊 dataset_loader.py      # TuSimple dataset handling
├── 🎯 train.py               # Training script with full pipeline
├── 👁️ test_loader.py         # Visualization and testing utilities
├── 📁 tusimple/              # TuSimple dataset directory
│   └── TUSimple/
│       ├── train_set/        # Training images and annotations
│       └── test_set/         # Testing images and annotations
└── 💾 model_epoch_*.pth      # Saved model checkpoints
```

## 🧠 Model Details

### DoubleConv Block

The fundamental building block of our U-Net:

```python
class DoubleConv(nn.Module):
    """
    Performs: Conv2d → ReLU → Conv2d → ReLU
    
    Benefits:
    - Feature extraction at each resolution level
    - Maintains spatial dimensions (padding=1)
    - Non-linear activation for complex pattern learning
    """
```

**Key Characteristics**:
- **Kernel Size**: 3×3 (optimal for local feature extraction)
- **Padding**: 1 (preserves spatial dimensions)
- **Activation**: ReLU (prevents vanishing gradients)
- **Memory Optimization**: inplace=True for ReLU

### SimpleUNet Architecture

#### Layer-by-Layer Breakdown

| Layer | Input Shape | Output Shape | Purpose |
|-------|-------------|--------------|---------|
| `enc1` | (B, 3, 256, 512) | (B, 32, 256, 512) | Extract low-level features |
| `pool` | (B, 32, 256, 512) | (B, 32, 128, 256) | Reduce spatial dimensions |
| `enc2` | (B, 32, 128, 256) | (B, 64, 128, 256) | Extract high-level features |
| `up` | (B, 64, 128, 256) | (B, 32, 256, 512) | Restore spatial dimensions |
| `concat` | [(B, 32, 256, 512), (B, 32, 256, 512)] | (B, 64, 256, 512) | Combine features |
| `dec1` | (B, 64, 256, 512) | (B, 32, 256, 512) | Process combined features |
| `outc` | (B, 32, 256, 512) | (B, 1, 256, 512) | Generate final prediction |

*B = Batch Size*

#### Skip Connection Benefits

1. **🔍 Gradient Flow**: Enables training of deeper networks
2. **📍 Spatial Preservation**: Maintains fine-grained location information
3. **🎯 Multi-Scale Features**: Combines low-level and high-level representations
4. **⚡ Faster Convergence**: Accelerates training process

## 📚 Code Walkthrough

### 1. Dataset Loading (`dataset_loader.py`)

#### Key Components:

**TuSimpleDataset Class**:
```python
class TuSimpleDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        # Loads JSON annotations and discovers image files
        
    def __getitem__(self, idx):
        # 1. Load image from file path
        # 2. Extract lane coordinates from JSON
        # 3. Generate binary mask from coordinates
        # 4. Apply preprocessing transforms
        # 5. Return (image, mask) pair
```

**Mask Generation Process**:
1. **Initialize**: Create zero-filled mask matching image dimensions
2. **Iterate**: Process each lane's coordinate points
3. **Draw**: Use `cv2.circle()` to create thick lane lines (radius=2)
4. **Convert**: Transform mask alongside image for consistency

**Transform Pipeline**:
- `ToPILImage()`: Convert numpy arrays to PIL format
- `Resize((256, 512))`: Standardize input dimensions
- `ToTensor()`: Convert to PyTorch tensors and normalize [0,1]

### 2. Model Architecture (`model.py`)

#### DoubleConv Implementation:
```python
def forward(self, x):
    # Sequential execution:
    # x → Conv2d → ReLU → Conv2d → ReLU → output
    return self.net(x)
```

#### U-Net Forward Pass:
```python
def forward(self, x):
    # Encoder path
    e1 = self.enc1(x)           # Low-level features
    e2 = self.enc2(self.pool(e1))  # High-level features
    
    # Decoder path
    d1 = self.up(e2)            # Upsample
    d1 = torch.cat([d1, e1], dim=1)  # Skip connection
    out = self.outc(self.dec1(d1))   # Final prediction
    
    return out  # Raw logits (apply sigmoid for probabilities)
```

### 3. Training Pipeline (`train.py`)

#### Training Loop Components:

**Data Loading**:
```python
for imgs, masks in train_loader:
    imgs = imgs.to(torch.float32)   # Ensure correct dtype
    masks = masks.to(torch.float32) # Match network expectations
```

**Forward Pass**:
```python
preds = model(imgs).squeeze(1)  # Remove channel dim for BCE loss
loss = criterion(preds, masks)  # Compute binary cross-entropy
```

**Backpropagation**:
```python
optimizer.zero_grad()  # Clear previous gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update parameters
```

**Checkpointing**:
```python
torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
```

### 4. Visualization (`test_loader.py`)

#### Sample Display Process:
1. **Load**: Single batch from DataLoader
2. **Convert**: Tensor format to numpy for matplotlib
3. **Display**: Side-by-side image and mask visualization
4. **Format**: RGB image + grayscale binary mask

## 🔧 Training Process

### Hyperparameter Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 4 | Balance between memory usage and gradient stability |
| **Learning Rate** | 1e-4 | Conservative rate for stable convergence |
| **Epochs** | 10 | Initial training; monitor for early stopping |
| **Optimizer** | Adam | Adaptive learning rates, good for computer vision |
| **Loss Function** | BCEWithLogitsLoss | Numerical stability + binary classification |

### Loss Function Deep Dive

**BCEWithLogitsLoss Benefits**:
- **🧮 Numerical Stability**: Combines sigmoid + BCE in single operation
- **⚖️ Class Balance**: Handles imbalanced lane/background pixels
- **📊 Probabilistic Output**: Provides confidence scores
- **🎯 Gradient Quality**: Smooth gradients for optimization

Mathematical formulation:
```
Loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
where σ(x) = 1/(1 + e^(-x))
```

### Training Monitoring

**Key Metrics to Track**:
- **📉 Loss Curves**: Training loss per epoch
- **⏱️ Training Time**: Batch processing speed
- **💾 Memory Usage**: GPU/CPU utilization
- **🎯 Convergence**: Loss stabilization indicators

**Expected Training Progression**:
1. **Epochs 1-3**: Rapid loss decrease (learning basic features)
2. **Epochs 4-7**: Moderate improvement (refining boundaries)
3. **Epochs 8-10**: Fine-tuning (convergence phase)

## 📈 Performance

### Evaluation Metrics

For lane detection tasks, consider these metrics:

**Pixel-Level Metrics**:
- **🎯 IoU (Intersection over Union)**: Overlap between predicted and ground truth
- **📊 F1-Score**: Harmonic mean of precision and recall
- **✅ Accuracy**: Correct pixel classifications

**Lane-Level Metrics**:
- **📏 Lane Detection Rate**: Percentage of correctly detected lanes
- **📐 Position Accuracy**: Distance error in lane center estimation
- **🚗 False Positive Rate**: Incorrectly detected lane segments

### Optimization Strategies

**Model Improvements**:
1. **🔧 Architecture**: Add more encoder/decoder levels
2. **📊 Data Augmentation**: Rotation, brightness, contrast variations
3. **⚖️ Loss Functions**: Focal loss for class imbalance
4. **🎯 Post-processing**: Morphological operations for cleaner masks

**Performance Tuning**:
1. **⚡ Mixed Precision**: Use float16 for faster training
2. **📱 Model Pruning**: Remove redundant parameters
3. **🔄 Knowledge Distillation**: Transfer from larger models
4. **📊 Quantization**: Int8 inference for deployment

## 🛠️ Customization

### Adapting for Different Datasets

**Dataset Integration Steps**:
1. **📊 Format Analysis**: Understand annotation structure
2. **🔄 Loader Modification**: Adapt `__getitem__` method
3. **📏 Resolution Adjustment**: Update image dimensions
4. **🎨 Visualization**: Modify display functions

**Example: CULANE Dataset**:
```python
class CULaneDataset(Dataset):
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_paths[idx])
        
        # Load segmentation mask (different format)
        mask = cv2.imread(self.mask_paths[idx], 0)  # Grayscale
        
        # Apply same transforms
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            
        return img, mask
```

### Model Architecture Variations

**Enhanced U-Net Options**:
1. **🏗️ Deeper Networks**: More encoder/decoder levels
2. **🔀 Attention Mechanisms**: Focus on relevant features
3. **📊 Batch Normalization**: Improve training stability
4. **🎯 Dense Connections**: DenseNet-style feature reuse

**Alternative Architectures**:
- **🚀 DeepLab**: Atrous convolutions for multi-scale features
- **⚡ ENet**: Lightweight architecture for real-time inference
- **🎯 SCNN**: Spatial CNN for lane-specific features

### Deployment Considerations

**Production Optimization**:
1. **📱 Model Conversion**: ONNX, TensorRT, Core ML
2. **⚡ Inference Speed**: Batch processing, async execution
3. **💾 Memory Management**: Efficient tensor operations
4. **🔧 Error Handling**: Robust input validation

**Integration Example**:
```python
class LaneDetector:
    def __init__(self, model_path):
        self.model = SimpleUNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def detect_lanes(self, image):
        # Preprocess input
        processed = self.preprocess(image)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(processed)
            
        # Post-process output
        lanes = self.postprocess(prediction)
        return lanes
```

## 📖 References

### Academic Papers
- **U-Net**: Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- **TuSimple**: TuSimple. "Towards End-to-End Lane Detection: An Instance Segmentation Approach." ArXiv 2018.

### Technical Resources
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/)
- **Computer Vision**: [OpenCV Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- **Deep Learning**: [Deep Learning Book](https://www.deeplearningbook.org/)

### Dataset Resources
- **TuSimple Benchmark**: [GitHub Repository](https://github.com/TuSimple/tusimple-benchmark)
- **CULANE Dataset**: [CULANE](https://xingangpan.github.io/projects/CULane.html)
- **BDD100K**: [Berkeley DeepDrive](https://bdd-data.berkeley.edu/)

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **🔧 Model architectures**: Implement attention mechanisms
- **📊 Evaluation metrics**: Add comprehensive benchmarking
- **🎨 Visualization**: Enhanced result display
- **📚 Documentation**: Tutorial notebooks and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TuSimple** for providing the benchmark dataset
- **U-Net authors** for the foundational architecture
- **PyTorch community** for excellent deep learning framework
- **OpenCV** for computer vision utilities

---

<div align="center">

**Built with ❤️ for autonomous vehicle research**

[⭐ Star this repo](https://github.com/yourusername/lane-detection) • [🐛 Report issues](https://github.com/yourusername/lane-detection/issues) • [💡 Request features](https://github.com/yourusername/lane-detection/pulls)

</div>

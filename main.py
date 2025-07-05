import torch
from ultralytics import YOLO
import os

def main():
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # if torch.backends.mps.is_available():
    #     print("Using Apple Silicon GPU (MPS backend)")
    #     device = "mps"
    # else:
    #     print("Using CPU")
    #     device = "cpu"

    device = "cpu"

    model = YOLO("yolov8m-seg.pt")  # Load a pre-trained YOLOv8 model

    # Training configuration for lane detection
    results = model.train(
        data='dataset/dataset.yaml',  # Your dataset configuration
        epochs=100,
        imgsz=640,
        batch=8,  # Reduced batch size for medium model on MPS
        device=device,
        project='lane_detection',
        name='yolov8m_seg',
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        workers=2,
        amp=False,  # Disable AMP for MPS compatibility
        optimizer='AdamW',
        lr0=0.001,
    )
    
    return results

if __name__ == "__main__":
    main()
    print("Training completed.")
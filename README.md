# LicensePlateRecognitionSystem
YOLOv8 Instance Segmentation - Custom Cat Plate Detection (my_pla2) This project focuses on training a custom instance segmentation model using Ultralytics YOLOv8 to detect and segment special vehicle license plates labeled as my_pla2. The dataset was prepared using CVAT, and the model was trained and tested on Google Colab using a Tesla T4 GPU.

Project Structure
.
├── my_pla2_data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── test_image_folder.zip
├── test_image_folder/
├── yolov8n-seg.pt               # Pretrained YOLOv8 segmentation model
├── runs/
│   └── segment/
│       └── train/               # Training outputs
├── notebook.ipynb               # Colab training and inference notebook

Features
-Custom instance segmentation using YOLOv8n-seg
-Single-class segmentation (my_pla2)
-Trained on a dataset of annotated images
-Uses Albumentations for advanced augmentation
-Achieves high accuracy and fast inference
-Inference tested successfully on new images

Model Performance (Validation Set)
Metric	Value
Precision	1.000
Recall	1.000
mAP@0.5	0.995
mAP@0.5:0.95	0.94

How to Use
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/my_pla2-segmentation.git
cd my_pla2-segmentation
2. Install Dependencies
bash
Copy
Edit
pip install ultralytics opencv-python albumentations
3. Run Inference on a New Image
python
Copy
Edit
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/segment/train/weights/best.pt")

# Run prediction on a test image
results = model.predict(source="test_image_folder/OZEL_PLAKA_02.jpg", save=True)
Results are saved to runs/segment/predict.

Dataset Details
Annotated using CVAT (in YOLOv8 segmentation format)

Structure:

images/ and labels/ directories for train, valid, and test sets

One class: my_pla2

The data.yaml file defines paths and class info

Training Configuration
Model: yolov8n-seg.pt (YOLOv8 nano segmentation)

Epochs: 10

Image size: 640

Optimizer: AdamW (auto-selected by Ultralytics)

AMP (mixed precision): Enabled

Data Augmentations:

CLAHE

Gaussian & Median Blur

Random grayscale

Horizontal flip

Results Visualization
Visual outputs are saved under:

bash
Copy
Edit
runs/segment/train/
runs/segment/predict/
You can find training metrics, loss curves, and prediction images in those folders.


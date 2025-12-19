<div align="center">

📊 Chart Scale Detector (Sim2Real)

Automated Axis Extraction for XANES & Infrared Spectroscopy


<!-- <img src="docs/demo_result.jpg" width="800" alt="Inference Result"> -->

<p align="center">
<strong>A Sim2Real solution for digitizing scientific charts using YOLOv11 Pose estimation.</strong>
</p>

Overview • Workflow • File Structure • Usage

</div>

📖 Project Overview

This project addresses the challenge of automatically extracting chart data (specifically axes and ticks) from scientific literature, such as XANES and Raman spectra.

Traditional OCR often fails on complex, low-quality scientific plots. To overcome the scarcity of real annotated data, this project employs a Sim2Real (Simulation to Reality) strategy:

Synthetic Generation 🎨: Batch-generating diverse charts with Matplotlib.

Domain Adaptation 🌫️: "Corroding" synthetic images (blur, noise, compression) to mimic scanned documents.

Pose Estimation 🧩: Using YOLOv11-Pose to detect axes and automatically pair Tick Marks with Tick Labels.

🚀 Workflow

The pipeline consists of data generation, augmentation, mixed training, and inference.

graph LR
    subgraph "1. Data Prep (Sim2Real)"
    A[Gen: Synthetic Charts] -->|Matplotlib| B(Clean Images)
    B -->|Augment: Blur/Noise| C(Degraded Images)
    D[Real Annotations] -->|json2yolo| E(Real Images)
    end
    
    subgraph "2. Training"
    C & E -->|Split & Mix| F{Mixed Dataset}
    F -->|Stage 1: Pre-train| G[YOLOv11 Pose Model]
    G -->|Stage 2: Fine-tune| H[Final Model]
    end
    
    subgraph "3. Deployment"
    H -->|Inference| I[Axis & Tick Extraction]
    end
    
    style H fill:#f96,stroke:#333,stroke-width:2px


📂 File Descriptions

1. Data Generation & Augmentation

Scripts for creating the "Fake" dataset that looks "Real".

File

Description

synthetic chart generator.py

Core Engine. Batch-generates charts (line/scatter) and auto-creates YOLO Pose labels.

augment_data.py

Sim2Real Adapter. Applies Gaussian blur, JPEG compression, and noise to bridge the domain gap.

json2yolo_mixed.py

Format Converter. Converts real LabelMe JSONs to YOLO format, pairing scale points with text boxes.

2. Dataset Construction

Scripts for assembling the training data.

File

Description

1_split_data.py

Assembler. Mixes synthetic and real data, applies oversampling to real samples, and generates .yaml files.

synthetic chart verification.py

Debugger. Visualizes generated YOLO labels on synthetic images to verify coordinate accuracy.

val annotated real...yolo.py

Validator. Visualizes ground truth annotations on the validation set.

3. Model Training

Training pipelines using Ultralytics YOLOv11.

File

Description

2_train.py

Stage 1 (Pre-training). Trains on massive synthetic data (1024sz) using AdamW optimizer.

resume_train.py

Stage 2 (Fine-tuning). Transfer learning on the Mixed Dataset with aggressive augmentation (Mixup, Rotation).

4. Inference & Evaluation

Testing the model on real-world scientific papers.

File

Description

3_inference.py

Main Inference. Predicts on test images, separates X/Y axes, and visualizes tick-label connections.

real chart test...py

Legacy Test. Early version script for testing Top-2 anchor logic.

🛠️ Getting Started

Prerequisites

pip install ultralytics albumentations matplotlib opencv-python tqdm


Quick Usage

1. Generate Data

# Generate clean synthetic charts
python "synthetic chart generator.py"

# Apply Sim2Real degradation
python augment_data.py


2. Prepare Dataset

# Mix synthetic and real data
python 1_split_data.py


3. Train

# Start training (Stage 1 or Stage 2)
python resume_train.py


4. Inference

# Run detection on your images
python 3_inference.py


📊 Results Visualization

The model outputs bounding boxes for axis text and keypoints for tick marks.

Feature

Visualization Key

X-Axis

🟦 Blue Box

Y-Axis

🟥 Red Box

Tick Mark

🟢 Green Dot

Tick Label

🟡 Yellow Dot

Note: The line connecting the Green and Yellow dots represents the specific pairing predicted by the Pose model.

<div align="center">
<p>Developed for Scientific Data Extraction | 2025</p>
</div>

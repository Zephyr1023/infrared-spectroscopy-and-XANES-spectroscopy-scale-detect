<div align="center">

# 📊 Scientific Chart Axis Detector
### Sim2Real: XANES & Infrared Spectroscopy Auto-Digitization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11--Pose-orange?style=for-the-badge&logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>Automated chart data extraction using YOLOv11-Pose with Domain Adaptation.</strong>
</p>

[Project Overview](#-project-overview) • [Workflow](#-workflow) • [File Structure](#-file-structure) • [Installation](#-installation)

</div>

---

## 📖 Project Overview

This project addresses the challenge of automatically extracting chart data from scientific literature (e.g., **XANES**, **Raman spectra**). Traditional OCR methods often fail on complex chart axes. To overcome the scarcity of real annotated data, we employ a **Sim2Real (Simulation to Reality)** strategy:

| Strategy | Description |
| :--- | :--- |
| **1. Synthetic Generation** 🎨 | Batch-generate synthetic charts with precise annotations using Matplotlib. |
| **2. Domain Adaptation** 🌫️ | Apply **"corrosion"** (blurring, noise, compression) via Albumentations to simulate real scanned documents. |
| **3. Mixed Training** 🔄 | Train **YOLOv11-Pose** using massive synthetic data + limited oversampled real data. |
| **4. Keypoint Pairing** 🔗 | Automatically pairs **"tick marks"** with **"tick labels"** for precise coordinate mapping. |

---

## 🚀 Workflow

```mermaid
graph LR
    subgraph Data_Preparation
    A[Synthetic Generator] -->|Create| B(Pure Synthetic Images)
    B -->|Augment Data| C(Degraded Images)
    D[Manual Annotation] -->|json2yolo| E(Real Images & Labels)
    end
    
    subgraph Dataset_Building
    C & E -->|Split Data| F{Mixed Dataset}
    end
    
    subgraph Training_Inference
    F -->|Train/Resume| G[YOLOv11 Pose Model]
    G -->|Inference| H[Result Visualization]
    end
    
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px

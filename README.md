# infrared-spectroscopy-and-XANES-spectroscopy-scale-detect
Chart scale detector trained using YOLOv11L-Pose for infrared spectroscopy and XANES spectroscopy

Project Overview
This project aims to address the challenge of automatically extracting chart data from scientific literature (e.g., XANES, Raman spectra). To overcome the scarcity of real annotated data, it employs a Sim2Real (Simulation to Reality) strategy:
1. Synthetic Data Generation: Batch-generate synthetic charts with precise annotations using Matplotlib.
2. Domain Adaptation: Apply “corrosion” processing (blurring, noise, compression) to synthetic data via Albumentations, simulating the texture of real scanned documents.
3. Mixed Training: Train the YOLOv11-Pose model using a large volume of synthetic data combined with a small amount of oversampled, real annotated data.
4. Keypoint Detection: The model not only detects axis elements but also automatically pairs “tick marks” with “tick labels” through a Keypoint structure, providing a core reference for subsequent coordinate mapping.

File Descriptions
1. Data Generation & Augmentation
synthetic chart generator.py Core generation engine that uses Matplotlib to batch-generate synthetic charts in various styles, including line charts and scatter plots, while automatically producing corresponding images and YOLO Pose format labels.
augment_data.py The critical step in Sim2Real, using Albumentations to apply Gaussian blur, JPEG compression, and noise addition to synthetic high-resolution charts. This reduces domain differences between synthetic data and real scanned documents.
json2yolo_mixed.py Data format conversion tool that transforms LabelMe JSON annotations from real data into YOLO Pose format. It automatically pairs scale points (Point) with numerical boxes (Box) into keypoint groups using distance algorithms.
2. Dataset Construction
1_split_data.py Dataset assembly script responsible for proportionally blending “corroded synthetic data” with “real data,” applying oversampling strategies to real data, and ultimately generating the required directory structure and .yaml configuration files for training.
synthetic chart verification.py Synthetic data quality inspection tool. Re-draws generated YOLO labels onto synthetic images to verify the accuracy of the generator's coordinate calculations.
val annotated real chart visualize_yolo.py Real data quality inspection tool. Visualizes the annotated data in the validation set to ensure the logic of keypoints and bounding boxes converted from manual annotations is correct.
3. Model Training
2_train.py First-stage training script. Pre-trains the YOLOv11-Pose model using extensive synthetic data, employing the AdamW optimizer and high-resolution (1024px) inputs to capture fine-grained details.
resume_train.py Second-stage (fine-tuning) training script. Loads pre-trained weights for transfer learning on a mixed dataset (Real + Synthetic). Employs more aggressive data augmentation (e.g., Mixup, rotation) to enhance generalization.
4. Inference & Evaluation
3_inference.py Primary inference script. Loads the final trained model to predict test set images, automatically separates X/Y axis data, plots lines connecting scales and values, and outputs visual results.
real chart test(second training).py An earlier version of the inference testing script, primarily used to evaluate the performance of models trained in the second training phase (training set: synthetic charts; validation set: real spectra). It includes basic random sampling and result filtering functionality.

graph TD
    A[synthetic chart generator] --> B(Synthetic Images)
    B -->|augment_data| C(Degraded Images)
    D[LabelMe Manual Annotation] -->|json2yolo_mixed| E(Real Images & Labels)
    C & E -->|1_split_data| F{Mixed Dataset}
    F -->|2_train / resume_train| G[YOLOv11 Pose Model]
    G -->|3_inference| H[Result Visualization]

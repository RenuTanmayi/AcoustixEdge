# Acoustix-Edge Compass: Real-Time Acoustic Localization

An Edge-AI system designed for the hearing impaired to classify and localize safety-critical sounds in real-time. This project implements a high-accuracy ensemble of lightweight convolutional neural networks optimized for deployment on **Raspberry Pi 5**.

## 🚀 Key Features
* **Joint Classification & Localization:** Detects 13 distinct categories including emergency sirens and vehicle horns with directional labels (Left, Right, Middle).
* **Edge-Optimized Ensemble:** Combines **EfficientNet-Lite0**, **MobileNetV3**, and **GhostNet** to achieve robust predictions.
* **High Performance:** Achieved **89.0% Accuracy** on a 13-class spatial acoustic dataset (approx. 11x improvement over random baseline).
* **Privacy-First:** Designed for local inference where audio data is processed entirely on the edge device.

## 🧠 Technical Implementation
The project leverages the `timm` (PyTorch Image Models) library to ensemble three state-of-the-art architectures. By using a **Soft-Voting Ensemble**, the system reduces variance and improves reliability in noisy urban environments.

* **Input:** 3-channel Mel-spectrograms.
* **Backbones:** EfficientNet-Lite0, MobileNetV3-Small, GhostNet-100.
* **Optimization:** Stochastic Gradient Descent with Cross-Entropy Loss.

## 📊 Evaluation
The ensemble model produces the following results across 13 categories (Ambulance, FireTruck, PoliceCar, CarHorns, and Noise):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 89% |
| **Classes** | 13 |
| **Baseline** | 7.7% |

## 🛠️ Current Roadmap
- [x] Initial Model Training & Ensemble Logic
- [x] Achievement of 88% Accuracy benchmark
- [ ] Transitioning to **ONNX Runtime** for hardware acceleration
- [ ] Deployment on **Raspberry Pi 5** with real-time audio input

## 📄 Attributions
This project utilizes a standardized spatial-acoustic dataset for training and validation of the Direction-of-Arrival (DoA) logic.
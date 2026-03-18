Acoustix-Edge Compass
Real-Time Sound Detection and Direction for the Hearing Impaired
Acoustix-Edge is a wearable AI system designed to help hearing-impaired individuals stay safe. It listens for 13 specific sounds (like sirens and horns) and tells the user both what the sound is and where it is coming from (Left, Right, or Middle).

Project Overview
Hardware Target: Raspberry Pi 5 (optimized for edge computing).

Current Performance: 89% Accuracy across 13 categories.

Privacy: All audio processing happens locally on the device; no data is ever uploaded or stored.

How it Works
The system uses an Ensemble approach, which means it asks three different specialized AI models to "vote" on a sound to ensure the highest possible reliability.

Audio to Visual: Raw sound is converted into a Mel-spectrogram (a "picture" of the sound's frequency).

AI Analysis: Three lightweight models (EfficientNet-Lite0, MobileNetV3, and GhostNet) analyze the spectrogram simultaneously.

Smart Voting: The system averages their predictions to give a final result. This "Soft-Voting" method is why the accuracy is so high (89%) compared to a random guess (7.7%).

Sound Categories
The model recognizes 13 distinct event-direction pairs:

Ambulance: Left, Middle, Right

Fire Truck: Left, Middle, Right

Police Car: Left, Middle, Right

Car Horns: Left, Middle, Right

Background Noise (to prevent false alarms)

Tech Stack
Language: Python

Deep Learning: PyTorch, TIMM (PyTorch Image Models)

Processing: NumPy, Scikit-learn, Pillow

Hardware: Raspberry Pi 5

Roadmap
[x] Train 3-model ensemble on 13-class dataset.

[x] Achieve 89% validation accuracy benchmark.

[ ] Optimize for ONNX Runtime to reduce latency on the Pi 5.

[ ] Integrate haptic feedback (vibration) for directional alerts.
# Multi-Channel 3DGS Reconstruction with RGB-D Input

This repository is a customized extension of [SGS-SLAM](https://github.com/YourReference/SGS-SLAM), designed for multi-channel 3D Gaussian Splatting (3DGS) reconstruction with both semantic and affordance information. The system takes RGB-D sequences as input and builds an enhanced scene representation through a modular pipeline.

## 🧠 Overview

### 🔧 Based on:
- [SGS-SLAM](https://github.com/YourReference/SGS-SLAM) — a real-time SLAM system for 3D Gaussian Splatting.

### 📥 Input:
- RGB-D image sequences (color + aligned depth).

### 🔄 Pipeline:
1. **Pose Estimation Module**  
   Estimates camera poses from the RGB-D sequence.

2. **Multi-Channel 3DGS Reconstruction**  
   After pose estimation, the system reconstructs a 3D Gaussian map with multiple semantic channels, including:
   - **Semantic segmentation**
   - **Affordance segmentation**  
   These are encoded into different attribute channels of the 3D Gaussians.


---

## ✅ TODO List

- [ ] Integrate open-vocabulary detection module (e.g., Grounding DINO, OWL-ViT)
- [ ] Decouple and modularize the tracking (pose estimation) module
- [ ] Add and refine affordance segmentation channel in 3DGS

---

## 📌 Notes

- This repo is under active development.  
- We aim for modularity, allowing plug-and-play components for detection, tracking, and rendering.

---

## 📄 License

This project inherits the license of SGS-SLAM (check original repo for details).

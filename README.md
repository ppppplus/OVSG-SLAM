# Multi-Channel 3DGS Reconstruction with RGB-D Input

This repository is a customized extension of [SGS-SLAM](https://github.com/YourReference/SGS-SLAM), designed for **multi-channel 3D Gaussian Splatting (3DGS)** reconstruction with both semantic and affordance information. The system takes RGB-D sequences as input and builds an enhanced scene representation through a modular pipeline.

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
   - **Open-vocabulary object features (via Detic)**  
     These are encoded into different attribute channels of the 3D Gaussians.

---

## ⚙️ Environment Setup

   ```bash
   conda create -n sgs-slam python=3.9
   conda activate sgs-slam
   conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cudatoolkit=11.8 pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```
For Detic’s open-vocabulary features, please follow the official installation instructions:
👉 [Detic Installation Guide](https://github.com/facebookresearch/Detic/tree/main)
After creating the conda environment above, install the required packages listed in Detic’s documentation (e.g., Detectron2, fvcore, iopath, pycocotools, etc.).

---

## 🚀 Running on Replica Dataset

### 0. Download pre-trained models
- Detic models
   ```bash
   cd encoder/detic_encoder
   mkdir models
   wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
   ```
   Download more pre-trained models [here](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md#real-time-models) and put them into `encoder/detic_encoder/models`.

- Feature compression model
   Download from [Google Drive](https://drive.google.com/drive/folders/1rRhz_PoTIwkmJ2Tyfua3tW0wbo_re6W7?usp=sharing) and put it under `encoder/feat_comp/`.
### 1. Preprocess the Replica data:

```bash
python -m preprocess.replica.replica_preprocess
```

### 2. Run the SLAM system:

```bash
python scripts/slam.py configs/replica/slam.py
```

### 3. Run the post-optimization module:

```bash
python scripts/post_slam_opt.py configs/replica/post_slam_opt.py
```

### 4. Run the afford-transfer module:

```bash
python scripts/post_slam_opt_afford.py configs/replica/post_slam_opt_afford.py
```

### 5. Visualize the reconstruction online:

```bash
python viz_scripts/online_recon.py configs/replica/slam.py
```

---

## ✅ TODO List

- [x] Integrate open-vocabulary detection module (Detic)
- [x] Decouple and modularize the tracking (pose estimation) module
- [ ] Add and refine affordance segmentation channel in 3DGS

---

## 📌 Notes

- This repo is under active development.  
- The system is modular by design, allowing **plug-and-play components** for detection, tracking, and rendering.

---

## 📄 License

This project inherits the license of SGS-SLAM (check the original repo for details).

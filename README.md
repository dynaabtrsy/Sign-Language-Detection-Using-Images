# Sign Language Detection Using Images 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-000000?style=for-the-badge&logo=yolo)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-0097A7?style=for-the-badge&logo=google)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)

> **Comparing Pixel-Based Object Detection (YOLOv5s) vs. Landmark-Based Pose Estimation (MediaPipe Hands + MLP) for Real-Time Alphanumeric Sign Language Recognition.**

---

## 📌 1. Overview

Communication is a fundamental human need, yet individuals with hearing and speech impairments often face severe communication barriers in a predominantly hearing society. Sign language—such as American Sign Language (ASL)—serves as a primary medium for interaction. 

This project explores automated **Sign Language Recognition (SLR)** by engineering and benchmarking two distinct computer vision paradigms:
1. **YOLOv5s (Pixel-Based CNN):** A single-stage, anchor-based object detector predicting bounding boxes and gesture classes directly from raw pixel grids.
2. **MediaPipe Hands + MLP (Landmark-Based DNN):** A geometric pose estimation pipeline extracting 21 3D hand joint coordinates $(X, Y, Z)$ to feed a lightweight Multi-Layer Perceptron (MLP).

Developed under the **Intelligent Systems (BICS 2303)** course at the **International Islamic University Malaysia (IIUM)**, this repository evaluates both prototypes in terms of recognition accuracy, computational efficiency, and robustness against environmental variations.

---

## 💡 2. Executive Summary

* **The Core Problem:** Deep learning models for sign language often struggle with background noise, extreme lighting shifts, similar hand shapes (e.g., distinguishing 'M' vs. 'N'), and high computational overhead on edge devices.
* **Dual-Prototype Approach:** We developed a pixel-based object detector (**YOLOv5s**) and a landmark-based structural classifier (**MediaPipe + MLP**) to benchmark their trade-offs.
* **YOLOv5s Peak Performance:** Scaling the training dataset from 50 to 2,000 images yielded an exceptional **98% Precision, 98% Recall, and 99% mAP@0.5:0.95**. However, it remained sensitive to background clutter and distant signs.
* **MediaPipe Efficiency:** MediaPipe completely eliminated background interference by mapping a 21-knuckle 3D landmark mesh, achieving **95.52% classification accuracy** within just 10 epochs while operating at significantly lower computational cost.
* **Core Takeaway:** While YOLOv5s excels at localized bounding box detection in controlled environments, **MediaPipe's structural coordinate abstraction proves far superior for lightweight, real-time sign language translation.**

---

## 🛠️ 3. Methodology

### Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
| PHASE I: Data Preprocessing & Stratified Sampling                                 |
| - Programmatically sampled 2,000 regional alphanumeric images (A-Z, 1-9)         |
| - Standardized RGB color spaces & resized matrices to 416x416 pixels               |
| - Normalized [0,1] floating points & validated bounding box coordinate syntax      |
+-----------------------------------------------------------------------------------+
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
+------------------------------------+       +------------------------------------+
| PARADIGM A: YOLOv5s (Pixel-Based)  |       | PARADIGM B: MediaPipe (Landmarks)  |
| - CSP-Backbone feature extraction  |       | - Hand Palm Detector               |
| - PANet Neck multi-scale fusion    |       | - Extracts 21 3D joint coordinates |
| - Grid-based box & class prediction|       | - Flattened tensor vector feed     |
+------------------------------------+       +------------------------------------+
                   │                                           │
                   ▼                                           ▼
+------------------------------------+       +------------------------------------+
| Loss: Binary Cross-Entropy + CIoU  |       | Classifier: Multi-Layer Perceptron |
| Optimization: Transfer Learning    |       | Training: Rapid 10-Epoch          |
| Weights: COCO Pre-trained (MS)     |       | Convergence                        |
+------------------------------------+       +------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| PHASE IV: Comparative Benchmark & Real-Time GUI Evaluation                        |
| - Precision, Recall, mAP@0.5, mAP@0.5:0.95 vs. Categorical Accuracy Metrics       |
| - Real-time OpenCV stream with bounding boxes vs. skeletal mesh overlays           |
+-----------------------------------------------------------------------------------+
```

### Mathematical Formulation
- **YOLOv5 Multi-Task Loss Function:**
  $$\text{Loss} = \lambda_1 L_{\text{cls}} + \lambda_2 L_{\text{obj}} + \lambda_3 L_{\text{loc}}$$
  *Where $L_{\text{cls}}$ is Binary Cross-Entropy loss, $L_{\text{obj}}$ is objectness confidence loss, and $L_{\text{loc}}$ utilizes Complete Intersection over Union (CIoU).*

---

## 🔑 4. Key Findings

### 1. The Impact of Data Scaling on YOLOv5s
Evaluating YOLOv5s across four distinct dataset scales revealed a direct relationship between sample volume and model stability:

| Dataset Scale | Epochs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Key Observation |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **50 Images** | 50 | 3.6% | 100%* | 24.0% | 24.0% | **False Completeness:** Model drew massive boxes over backgrounds. |
| **500 Images** | 50 | 60.0% | 46.0% | 40.0% | 40.0% | Bounding box coordination stabilized; realistic recall achieved. |
| **1,000 Images**| 50 | 58.0% | 78.0% | 83.0% | 83.0% | Significant reduction in validation classification loss. |
| **2,000 Images**| 100 | **98.0%** | **98.0%** | **99.0%** | **99.0%** | Optimal convergence with zero divergence spikes. |

*\*Note: 100% recall on 50 images represents "false completeness" where the model over-generalized bounding boxes to avoid missing targets.*

### 2. Paradigm Comparison: YOLOv5s vs. MediaPipe

| Feature / Metric | YOLOv5s (2,000-Sample Model) | MediaPipe + MLP Prototype |
| :--- | :--- | :--- |
| **Primary Input** | Raw 2D RGB Pixel Grids ($416\times416$) | 21 3D Landmark Coordinates $(X,Y,Z)$ |
| **Detection Rate** | 100% hand localization in validation batches | Focuses exclusively on joint morphology |
| **Classification Score**| 98% Precision / 99% mAP | **95.52% Test Accuracy** |
| **Background Sensitivity**| Moderate to High (vulnerable to clutter/lighting) | **Zero** (abstracts background completely) |
| **Training Time** | ~2 hours on T4 GPU | **Under 2 minutes** (10 epochs) |
| **Target Best Use** | Bounding box localization in controlled scenes | Real-time lightweight gesture classification |

---

## 🎯 5. Recommendations

1. **Adopt Pose Estimation for Mobile/Edge Deployment:**
   * *Action:* Utilize MediaPipe landmark extraction over full pixel-grid convolutions when deploying on low-power devices.
   * *Impact:* Reduces CPU/GPU overhead while offering high environmental robustness.
2. **Implement Focal Loss for Morphologically Similar Signs:**
   * *Action:* Replace standard Binary Cross-Entropy with Focal Loss in future YOLO iterations.
   * *Impact:* Mathematically penalizes "hard examples" (e.g., distinguishing letters 'M' and 'N') to eliminate fine-grained misclassifications.
3. **Transition to Dynamic Sequential Modeling (LSTMs / TSM):**
   * *Action:* Upgrade static image recognition pipelines to process video streams using Long Short-Term Memory (LSTM) units or Temporal Shift Modules (TSM).
   * *Impact:* Enables the system to understand continuous motion, transitioning from isolated alphabet signs to full word and sentence translation.
4. **Hardware-Aware Hyperparameter Optimization:**
   * *Action:* Cap image resolutions at $416\times416$ with batch sizes of 32 when training on cloud platforms like Google Colab T4.
   * *Impact:* Prevents GPU memory saturation, virtual RAM leaks, and unexpected session timeouts.

---

## ⚠️ 6. Limitations

* **Static Gesture Constraint:** The project is limited to static images/frames and cannot recognize dynamic sign language gestures requiring continuous motion over time.
* **Hardware & Cloud Limitations:** Session timeouts and RAM allocations on cloud platforms (e.g., Google Colab T4) constrained full-dataset training runs, necessitating stratified sub-sampling.
* **Fine-Grained Joint Ambiguity in YOLOv5s:** Bounding-box regression occasionally struggles to capture subtle finger overlap differences, relying too heavily on overall hand shape.

---

## 📈 7. Project Impact

* **Promoting Inclusivity:** Bridges the communication gap between hearing individuals and the Deaf/Hard-of-Hearing community through automated sign translation.
* **Benchmark Insights:** Provides empirical evidence on the trade-offs between raw pixel processing (CNNs) and anatomical landmark extraction (GNNs/MLPs) for accessibility software engineers.
* **Foundation for Educational Tools:** Serves as a modular codebase for building interactive sign language learning applications with real-time feedback.

---

## 📊 8. Dashboard & GUI Visualization

Both prototypes were integrated into interactive Python OpenCV interfaces for live testing.

🔗 **[Project Presentation & Prototype Demonstration Video](https://github.com/your-username/your-repo-name)** *(Replace with your link)*

### Prototype Real-Time Execution Highlights
1. **YOLOv5s Bounding Box Interface:** Displays precise bounding boxes around detected hands with categorical label tags and calculated confidence scores (e.g., `A 0.98`).
2. **MediaPipe 21-Knuckle Mesh Interface:** Renders an interconnected skeletal vector map directly over finger joints, performing real-time classification independent of background clutter.
3. **Sentence Generation Logic:** Integrated text-processing script that appends recognized alphanumeric characters into words, supporting `SPACE` and `DELETE` gestures.

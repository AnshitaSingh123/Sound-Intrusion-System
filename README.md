# Sound Classification for Intrusion Detection

## 📌 Project Overview

This project develops an environmental sound classification system in **MATLAB** for intrusion detection. The system identifies specific sound events from audio inputs, focusing on categories relevant to security applications. A user-friendly **Graphical User Interface (GUI)** allows users to select from various trained machine learning models for sound analysis.

---

## 🧹 Problem Statement

The goal is to accurately classify environmental sounds, specifically:

* `Gun Shot`
* `Siren`
* `Engine Idling`
* `Dog Bark`

using the **UrbanSound8K** dataset. The project uses both **traditional machine learning** and **deep learning models** to ensure robust classification performance, aiming for applications in automated surveillance and security systems.

---

## 📁 Dataset

This project uses the **[UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html)** — a collection of 8,732 labeled sound excerpts (≤ 4 seconds).

**Focused sound classes:**

* `gun_shot`
* `siren`
* `engine_idling`
* `dog_bark`

---

## 🗂️ Project Structure

```
Sound-Intrusion-System/
├── data/                       # UrbanSound8K dataset (audio + metadata)
├── models/                    # Trained ML/DL models (.mat files)
├── results/                   # Plots, metrics, etc. (optional)
├── src/
│   ├── feature_extraction/    # Scripts for MFCC, Spectrogram extraction
│   ├── preprocessing/         # Data loading, normalization, splitting
│   ├── training/              # SVM, RF, CNN, LSTM training scripts
│   ├── evaluation/            # Model evaluation scripts
│   └── gui/                   # MATLAB App Designer GUI files
└── main.m                     # (Optional) Orchestrates full workflow or launches GUI
```

---

## 🧠 Models Implemented

### 🧠 Traditional Machine Learning

* **Support Vector Machine (SVM):** Finds optimal decision boundaries.
* **Random Forest (RF):** Ensemble of decision trees, robust to noise.

### 🚀 Deep Learning

* **Convolutional Neural Network (CNN):** Extracts spatial features from spectrograms.
* **Long Short-Term Memory (LSTM):** Captures temporal dependencies.
* **(Optional Hybrid) CNN + LSTM:** CNN extracts features, LSTM models temporal patterns.

---

## 🛠️ Technical Stack

* **Language:** MATLAB R2023b (or newer)
* **Toolboxes:**

  * Deep Learning Toolbox
  * Audio Toolbox
  * Statistics and Machine Learning Toolbox

---

## 🚀 Getting Started

### ✅ Prerequisites

* MATLAB R2023b or newer
* Toolboxes mentioned above
* Download and extract the [UrbanSound8K dataset](https://urbansound8k.readthedocs.io/en/latest/)

  * Place the `audio/` and `metadata/` folders inside:

    ```
    data/UrbanSound8K/
    ```

---

### 🔧 Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AnshitaSingh123/Sound-Intrusion-System.git
   cd Sound-Intrusion-System
   ```

2. **Add Project to MATLAB Path**

   * Open MATLAB.
   * Navigate to the project root.
   
---

## 📊 Running the Project

### ↺ Step 1: Preprocess Data

```matlab
run src/preprocessing/preprocessAudio.m
```

* Loads audio files.
* Extracts MFCCs and spectrograms.
* Normalizes data and splits into train/val/test sets.
* Saves: `processed_data.mat`

---

### 🎯 Step 2: Train Traditional ML Models

```matlab
run src/training/trainSVM.m
run src/training/trainRandomForest.m
```

* Loads features.
* Trains SVM and Random Forest.
* Saves models to `models/` directory.

---

### 🤖 Step 3: Train Deep Learning Model

```matlab
run src/training/trainDeepLearningModel.m
```

* Trains CNN, LSTM, or hybrid model.
* Saves to `models/` directory.

---

### 💻 Step 4: Launch the GUI

```matlab
run src/gui/drdoproject.mlapp
```

* Select models.
* Load test audio.
* Visualize and analyze classification results.

---



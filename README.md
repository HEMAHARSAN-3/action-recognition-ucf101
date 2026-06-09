# 🎬 Action Recognition using Deep Learning (I3D)

## 📌 Project Overview

This project implements an **Action Recognition System** that classifies human actions from input videos using a deep learning-based architecture. The model processes a sequence of video frames and predicts the most probable action class from the **UCF101 dataset**, which contains 101 human activity categories.

The project provides a complete end-to-end pipeline including:

* Dataset preprocessing
* Video frame extraction
* Model training
* Model evaluation
* Checkpoint saving/loading
* Video inference
* Prediction of unseen videos

---

# 🚀 Features

* ✅ Deep Learning-based Action Recognition
* ✅ Supports the UCF101 Dataset (101 Classes)
* ✅ Automatic Video Frame Sampling
* ✅ Training Pipeline
* ✅ Evaluation Pipeline
* ✅ Checkpoint Management
* ✅ Inference on Custom Videos
* ✅ Modular Project Structure
* ✅ GPU (CUDA) Support

---

# 🧠 Model Architecture

The project uses an **Inflated 3D Convolutional Network (I3D)** architecture for spatio-temporal feature extraction.

The network learns:

* Spatial information from video frames
* Temporal information across consecutive frames
* Motion representation for action classification

Pipeline:

```
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
Resize & Preprocessing
      │
      ▼
I3D Network
      │
      ▼
Softmax Layer
      │
      ▼
Predicted Action
```

---

# 📂 Project Structure

```
action-recognition-ucf101/

│
├── configs/
│      ├── ucf101_i3d.yaml
│      └── ucf101_slowfast.yaml
│
├── data/
│      ├── dataset.py
│      ├── transforms.py
│      └── optical_flow.py
│
├── engine/
│      ├── trainer.py
│      ├── evaluator.py
│      ├── losses.py
│      └── scheduler.py
│
├── models/
│      ├── i3d.py
│      ├── slowfast.py
│      └── model_factory.py
│
├── scripts/
│      ├── train.py
│      ├── evaluate.py
│      ├── inference.py
│      ├── predict.py
│      └── extract_flow.py
│
├── checkpoints/
│      └── best_model.pth
│
├── outputs/
│      ├── metrics.json
│      ├── confusion_matrix.png
│      └── classification_report.json
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset

Dataset Used:

**UCF101 Action Recognition Dataset**

* 101 Action Categories
* Thousands of Real-world Videos
* Widely Used Benchmark for Video Classification

Example Classes:

* Archery
* Basketball
* Bowling
* HorseRace
* JumpRope
* PlayingGuitar
* Surfing
* TennisSwing
* VolleyballSpiking
* YoYo

and many more.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/HEMAHARSAN-3/action-recognition-ucf101.git

cd action-recognition-ucf101
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

Run:

```bash
python scripts/train.py \
--config configs/ucf101_i3d.yaml
```

The training pipeline will:

* Load Dataset
* Build Model
* Train Network
* Save Checkpoints
* Store Best Model

---

# 📈 Evaluation

Run:

```bash
python scripts/evaluate.py \
--config configs/ucf101_i3d.yaml \
--checkpoint checkpoints/best_model.pth
```

Outputs generated:

* metrics.json
* confusion_matrix.png
* classification_report.json

---

# 🎥 Inference on a Video

Run:

```bash
python scripts/inference.py \
--video path/to/video.mp4 \
--config configs/ucf101_i3d.yaml \
--checkpoint checkpoints/best_model.pth
```

Example Output:

```
===== PREDICTION =====

Class      : Bowling

Confidence : 1.66%
```

---

# 📌 Workflow

```
Dataset
      │
      ▼
Frame Sampling
      │
      ▼
Preprocessing
      │
      ▼
Model Training
      │
      ▼
Checkpoint Saving
      │
      ▼
Evaluation
      │
      ▼
Inference
      │
      ▼
Prediction
```

---

# 🛠 Technologies Used

* Python
* PyTorch
* NumPy
* OpenCV
* Scikit-learn
* YAML
* CUDA (GPU Support)
* Git
* GitHub

---

# 📌 Future Improvements

* Increase training epochs for better accuracy
* Hyperparameter tuning
* Data augmentation
* Real-time webcam action recognition
* Streamlit/Flask web interface
* ONNX/TensorRT deployment
* Mobile deployment

---

# 👨‍💻 Author

**Hema Harsan R**

GitHub:

https://github.com/HEMAHARSAN-3

---

# ⭐ Acknowledgement

This project was developed as part of an AI/ML assessment to demonstrate practical skills in:

* Artificial Intelligence
* Machine learning
* Deep Learning
* Computer Vision
* Video Understanding
* Model Training
* Evaluation
* Deployment Pipeline

If you find this project useful, please consider giving it a ⭐ on GitHub.

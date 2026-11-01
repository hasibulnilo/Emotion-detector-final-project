# Real-Time Facial Emotion Recognition
*****Live link: https://emotion-rayhan.streamlit.app/ 
A Python-based Facial Emotion Recognition (FER) project that detects and classifies human facial emotions using a trained Deep Learning model and OpenCV.

## 🚀 Project Overview

This project uses a trained CNN model to recognize human facial expressions in real time through a webcam.

The system detects faces and predicts the corresponding emotion using the trained `emotion.h5` model.

## ✨ Features

- Real-time face detection using OpenCV.
- Facial emotion recognition using a trained deep learning model.
- Webcam-based emotion detection.
- Haar Cascade-based face detection.
- Jupyter Notebook for model training and experimentation.

## 😊 Supported Emotions

The model recognizes five emotion classes:

- Happy
- Sad
- Angry
- Surprise
- Neutral

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Jupyter Notebook
- Deep Learning (CNN)

## 📁 Project Structure

```text
Emotion-Recognition/
│
├── images/
│
├── emotion.h5
├── haarcascade_frontalface_default.xml
│
├── realtimedetection.py
├── test.py
├── train_another_way.ipynb
│
├── req.txt
├── conda.txt
├── .gitignore
└── README.md
```

**Note:** Virtual environments, cache files, and generated build folders are excluded from GitHub.

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd Emotion-Recognition
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r req.txt
```

### 6. Run the Application

```bash
python realtimedetection.py
```

Make sure your webcam is connected and accessible.

## 🧠 Model Information

- Model file: `emotion.h5`
- Model type: Convolutional Neural Network (CNN)
- Task: Facial Emotion Classification
- Input: Facial image
- Output: Predicted emotion class

## 📓 Model Training

The Jupyter Notebook `train_another_way.ipynb` contains the model training and experimentation workflow.

To open the notebook:

```bash
jupyter notebook
```

Then open `train_another_way.ipynb`.

## 📦 Git Ignore Configuration

The following files and folders are excluded from Git tracking:

```gitignore
# Python cache
__pycache__/
*.py[cod]
*.pyo

# Jupyter Notebook
.ipynb_checkpoints/

# Virtual Environments
venv/
myenv/
tfenv/
transflow/

# Build files
build/
dist/

# IDE and Editor
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Environment variables
.env
.env.*

# Logs
*.log
```

## 🔄 Setup After Cloning

After cloning the repository, you do not need to upload or download an existing virtual environment.

Simply create a new environment and install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r req.txt
```

Your project will then be ready to run.

## 📌 Requirements

- Python 3.x
- Webcam
- Required Python libraries listed in `req.txt`
- Pre-trained emotion recognition model (`emotion.h5`)

## 👨‍💻 Author

**Hasibul Islam Niloy**

GitHub: [Your GitHub Profile](https://github.com/)

---

⭐ If you find this project useful, feel free to star the repository!
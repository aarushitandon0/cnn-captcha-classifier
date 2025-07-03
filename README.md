# CNN CAPTCHA Resolver

A deep learning project that builds a **Convolutional Neural Network (CNN)** to accurately classify **single-character CAPTCHA images**. It includes everything from dataset generation to training, evaluation, and prediction — built using TensorFlow and Keras.


##  Problem Statement

CAPTCHAs are used to distinguish humans from bots. This project solves **simple character-level CAPTCHAs** by training a CNN model from scratch on synthetically generated data. It learns to recognize **uppercase English letters and digits (A-Z, 0-9)** from distorted images.


## Features

- **Custom CAPTCHA dataset generator** using `captcha` Python library  
- **Image preprocessing pipeline**: grayscale, resize, normalize, reshape  
- **CNN model** with BatchNormalization, MaxPooling, and Dropout  
- **Training & validation with EarlyStopping**  
- **Evaluation** via accuracy, confusion matrix, classification report  
- **Misclassification visualization**  
- Predict on custom images


## Tech Stack

| Category        | Tool / Library |
|----------------|----------------|
|  Language | ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) |
|  Frameworks   | ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)<br>![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red?logo=keras) |
|  Image Proc.  | ![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green?logo=opencv) |
|  Math & Arrays | ![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-purple?logo=numpy) |
|  Plotting     | ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?logo=matplotlib) |
|  ML Tools     | ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn) |


##  How the Model Works
### Dataset Generation
- Synthetic CAPTCHA characters (A–Z, 0–9) generated using the captcha library and saved in class-wise folders.

### Preprocessing Pipeline

- Convert to grayscale

- Resize to (60x60)

- Normalize pixel values

- Add channel dimension (60,60,1)

### CNN Architecture

- Multiple Conv2D layers with ReLU and BatchNormalization

- MaxPooling2D for spatial reduction

- Dropout for regularization

- Final Dense layer with softmax for classification (36 classes)

### Training

- Optimizer: Adam

- Loss: Categorical Crossentropy

- EarlyStopping used to prevent overfitting

- Best Epoch performance: Epoch 9

✔️ Train Accuracy: 84.89%  
✔️ Train Loss:     0.4061  
✔️ Val Accuracy:   94.58%  
✔️ Val Loss:       0.1601  

- Achieved 94.58% test accuracy


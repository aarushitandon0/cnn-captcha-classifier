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

## Classification Report
Below is the performance breakdown per class (A–Z, 0–9) on the test set (2,880 samples):
```
              precision    recall  f1-score   support

           A       0.98      1.00      0.99       101
           B       0.82      0.99      0.89        81
           C       0.98      0.93      0.95        68
           D       0.87      0.96      0.91        74
           E       1.00      0.96      0.98        75
           F       0.96      1.00      0.98        79
           G       0.98      0.97      0.97        58
           H       0.98      0.99      0.98        84
           I       0.99      0.94      0.96        78
           J       0.96      1.00      0.98        87
           K       1.00      0.96      0.98        93
           L       0.98      0.97      0.97        90
           M       0.98      0.97      0.98        66
           N       1.00      0.96      0.98        90
           O       0.68      0.71      0.69        85
           P       0.96      1.00      0.98        86
           Q       1.00      0.99      0.99        85
           R       1.00      0.92      0.96        76
           S       1.00      0.83      0.90        86
           T       0.94      0.94      0.94        68
           U       0.96      0.99      0.97        69
           V       0.99      1.00      0.99        84
           W       0.95      0.99      0.97        77
           X       0.97      1.00      0.98        83
           Y       0.99      1.00      0.99        67
           Z       0.97      0.90      0.94        81
           0       0.72      0.64      0.68        87
           1       0.93      0.97      0.95        80
           2       0.91      0.99      0.95        76
           3       0.87      0.99      0.93        88
           4       0.99      1.00      0.99        77
           5       0.91      0.97      0.94        79
           6       1.00      0.99      0.99        78
           7       0.96      0.96      0.96        81
           8       1.00      0.77      0.87        88
           9       1.00      0.99      0.99        75

    accuracy                           0.95      2880
   macro avg       0.95      0.95      0.95      2880
weighted avg       0.95      0.95      0.95      2880

```

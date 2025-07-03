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


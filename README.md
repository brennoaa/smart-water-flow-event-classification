# Modelling individual residential water uses using machine learning algorithms 
This repository provides a dataset and a trained MLP-based model for non-intrusive classification of water consumption events from flow sensor time series, including preprocessing components and evaluation tools.The files correspond to the final model trained after expanding the training dataset.

---

## Objective

The objective of this work is to classify water consumption patterns based on smart sensor data, using Machine Learning techniques and signal preprocessing methods.

---

## Model

A Multilayer Perceptron (MLP) neural network was used with the following architecture:

- Input layer with 5 features: (i) the duration of the event, (ii) the median flow rate of the event, (iii) the flow rate decreasing time, the number of repetitions of the predominant event in the signal and (v) Number of times the event was divided into different samples.
- Dense layer with 32 neurons (ReLU)  
- Batch Normalization  
- Dropout (0.2)  
- Dense layer with 16 neurons (ReLU)  
- Output layer with Softmax activation and 8 classes: Flush toilet 1, Flush toilet 2, Flush toilet 3, Shower, Filter, Washing machine, Tap 1, Kitchen tap  

---

## Preprocessing

- Data normalization using `StandardScaler`  
- Class encoding using `LabelEncoder`  
- Train/test split: 70% / 30% 

---

## Model Evaluation

The model is evaluated using:

- Accuracy  
- Loss  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix  

---

## Project Files

- `modelo.h5` → trained model  
- `scaler.pkl` → data normalization  
- `encoder.pkl` → class encoding  
- `X_test.npy` → test data (features)  
- `y_test.npy` → test labels  
- `avaliar.py` → model evaluation script  

---

## How to Run

### 1. Install dependencies

pip install tensorflow scikit-learn numpy joblib

### 2. Place everything in a folder:

```bash
project/
│
├── model.h5
├── scaler.pkl
├── encoder.pkl
├── X_test.npy
├── y_test.npy
├── evaluate.py
```

### 3. Run evaluate.py


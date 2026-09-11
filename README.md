# Customer Churn Prediction using ANN

A Deep Learning project that predicts whether a bank customer is likely to **churn** or **stay** using an Artificial Neural Network (ANN).

## 📌 Project Overview

Customer churn means a customer stops using a company's services.

In this project, an Artificial Neural Network built using TensorFlow/Keras is trained on customer information to predict whether a customer is likely to leave the bank.

The project also includes a Streamlit web application for making predictions interactively.

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Matplotlib
* Seaborn
* Jupyter Notebook

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Encoding Categorical Data
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Build ANN
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Save Trained Model
   ↓
Streamlit Application
   ↓
Churn / No Churn Prediction
```

## 📂 Project Structure

```text
customer-churn-prediction-ann/
│
├── app.py
├── churn_modeling.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Churn_Modelling.csv
│
├── models/
│   └── churn_model.keras
│
└── images/
    └── app_screenshot.png
```

## 🤖 ANN Architecture

The Artificial Neural Network contains multiple Dense layers.

```text
Input Features
      ↓
Dense Layer - 6 neurons
      ↓
Dense Layer - 6 neurons
      ↓
Dense Layer - 5 neurons
      ↓
Dense Layer - 4 neurons
      ↓
Output Layer - 1 neuron
      ↓
Sigmoid Activation
      ↓
Churn / No Churn
```

### Activation Functions

* ReLU is used in the hidden layers.
* Sigmoid is used in the output layer because this is a binary classification problem.

## ⚙️ Data Preprocessing

The project uses:

* Label Encoding
* One-Hot Encoding
* StandardScaler
* Train/Test Split

## 📊 Model Training

The ANN is compiled using:

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metric: Accuracy
* Batch Size: 32
* Epochs: 15

## 🌐 Streamlit Application

The Streamlit application allows the user to enter customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card status
* Active Member status
* Estimated Salary

The trained ANN then predicts:

```text
Churn
```

or

```text
No Churn
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd customer-churn-prediction-ann
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Confusion Matrix

## 🎯 Project Goal

The goal of this project is to understand how an Artificial Neural Network can be used for a real-world binary classification problem and how a trained Deep Learning model can be integrated into a simple web application.

## 👨‍💻 Author

**Ajim Mujawar**

This project was created as part of my Deep Learning learning journey.

# ============================================================
# Artificial Neural Network - Customer Churn Prediction
# ============================================================

# ============================================================
# Part 1 - Data Preprocessing
# ============================================================

# Import libraries
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

np.random.seed(0)
tf.random.set_seed(0)


# ------------------------------------------------------------
# Import Dataset
# ------------------------------------------------------------

# Project structure:
# D:\ANN\
# └── churn-prediction-ann\
#     ├── data\
#     │   └── Churn_Modelling.csv
#     └── ...

BASE_DIR = Path(r"D:\ANN\churn-prediction-ann")
DATA_PATH = BASE_DIR / "data" / "Churn_Modelling.csv"

dataset = pd.read_csv(
    r"D:\ANN\churn-prediction-ann\data\Churn_Modelling.csv"
)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset shape:", dataset.shape)
print("\nFirst 5 rows:")
print(dataset.head())


# ------------------------------------------------------------
# Separate Independent and Dependent Variables
# ------------------------------------------------------------

# Columns 3 to 12 are used as input features
# Column 13 is the target variable: Exited

X = dataset.iloc[:, 3:13].copy()
y = dataset.iloc[:, 13].copy()

print("\nInput shape before encoding:", X.shape)
print("Target shape:", y.shape)


# ------------------------------------------------------------
# Create Dummy Variables
# ------------------------------------------------------------

# Geography:
# France, Germany, Spain
#
# drop_first=True removes one category to avoid redundant
# dummy variables.

geography = pd.get_dummies(
    X["Geography"],
    drop_first=True,
    dtype=int
)

# Gender:
# Female, Male

gender = pd.get_dummies(
    X["Gender"],
    drop_first=True,
    dtype=int
)


# ------------------------------------------------------------
# Concatenate DataFrames
# ------------------------------------------------------------

X = pd.concat(
    [X, geography, gender],
    axis=1
)


# ------------------------------------------------------------
# Drop Original Categorical Columns
# ------------------------------------------------------------

X = X.drop(
    ["Geography", "Gender"],
    axis=1
)

print("\nFeatures after encoding:")
print(X.columns.tolist())

print("\nFinal feature shape:", X.shape)


# ============================================================
# Part 1.5 - Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

# Fit ONLY on training data
X_train = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ============================================================
# Part 2 - Build the Artificial Neural Network
# ============================================================

print("\n" + "=" * 60)
print("BUILDING ARTIFICIAL NEURAL NETWORK")
print("=" * 60)


# ------------------------------------------------------------
# Initialize ANN
# ------------------------------------------------------------

classifier = tf.keras.Sequential()


# ------------------------------------------------------------
# Input Layer + First Hidden Layer
# ------------------------------------------------------------

classifier.add(
    tf.keras.layers.Dense(
        units=11,
        kernel_initializer="he_uniform",
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

# Dropout helps reduce overfitting
classifier.add(
    tf.keras.layers.Dropout(0.20)
)


# ------------------------------------------------------------
# Second Hidden Layer
# ------------------------------------------------------------

classifier.add(
    tf.keras.layers.Dense(
        units=11,
        kernel_initializer="he_normal",
        activation="relu"
    )
)

classifier.add(
    tf.keras.layers.Dropout(0.20)
)


# ------------------------------------------------------------
# Third Hidden Layer
# ------------------------------------------------------------

classifier.add(
    tf.keras.layers.Dense(
        units=15,
        kernel_initializer="he_normal",
        activation="relu"
    )
)


# ------------------------------------------------------------
# Output Layer
# ------------------------------------------------------------

classifier.add(
    tf.keras.layers.Dense(
        units=1,
        kernel_initializer="glorot_uniform",
        activation="sigmoid"
    )
)


# ------------------------------------------------------------
# Display Model Architecture
# ------------------------------------------------------------

classifier.summary()


# ============================================================
# Compile the ANN
# ============================================================

classifier.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# Part 2.5 - Early Stopping
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# Train the ANN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING ANN")
print("=" * 60)

model_history = classifier.fit(
    X_train,
    y_train,

    # 20% test data is completely separate.
    # validation_split creates validation data from training data.
    validation_split=0.20,

    batch_size=32,
    epochs=50,

    callbacks=[early_stopping],

    verbose=1
)


# ============================================================
# Part 2.6 - Training History
# ============================================================

print("\nTraining history keys:")
print(model_history.history.keys())


# ============================================================
# Accuracy Graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    model_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    model_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("ANN Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Loss Graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    model_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    model_history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Part 3 - Predictions
# ============================================================

print("\n" + "=" * 60)
print("MODEL PREDICTION")
print("=" * 60)


# Predict probability
y_pred_probability = classifier.predict(
    X_test,
    verbose=0
).ravel()


# ------------------------------------------------------------
# Convert probability into class
# ------------------------------------------------------------

# If probability > 0.5:
#     Customer churn = 1
#
# Otherwise:
#     Customer churn = 0

threshold = 0.50

y_pred = (
    y_pred_probability >= threshold
).astype(int)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# Display confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed (0)", "Churned (1)"]
).plot()

plt.title("ANN Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# Classification Report
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Stayed",
            "Churned"
        ]
    )
)


# ============================================================
# Evaluation Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_pred_probability
)


# ============================================================
# Final Results
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("=" * 60)


# ============================================================
# Save the Model
# ============================================================

MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)

classifier.save(
    MODEL_PATH / "churn_ann.keras"
)

print("\nModel saved successfully:")
print(MODEL_PATH / "churn_ann.keras")
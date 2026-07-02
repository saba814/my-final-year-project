# Heart Disease Prediction & Clinical Decision Support System

A comprehensive, production-ready machine learning framework and interactive clinical decision support application designed to predict the risk of cardiovascular diseases based on clinical health parameters. 

This project integrates a robust machine learning pipeline utilizing an **Artificial Neural Network (ANN)** alongside baseline models, paired with a modern **Streamlit** user interface and persistent local data storage.

---

## 📂 Project Repository Structure

The project encompasses the following key core files and components:

*   **`app.py`**: The primary dashboard application powered by Streamlit. It manages user input handling, displays risk metrics, provides interactive visualization placeholders, handles report compilation for clinical exports, and syncs with the database.
*   **`heart_model.py`**: The comprehensive backend pipeline module. It encapsulates automated data preprocessing (handling missing values, feature scaling), cross-validation, baseline modeling (Logistic Regression and Random Forest), and the deep learning implementation (TensorFlow/Keras ANN architecture).
*   **`Heart_Disease_Training_Workflow.ipynb`**: A well-documented Jupyter/Google Colab Notebook that acts as the model training notebook. It visualizes the end-to-end cloud training process, hyperparameter tuning steps, and model artifact exports.
*   **`prediction_history.db`**: A lightweight, standalone SQLite database designed to automatically track and persistently log patient data inputs, timestamps, and predicted risk results for audit trails.
*   **`processed.cleveland.data`**: The standard benchmark clinical dataset utilized for model validation, training, and algorithmic benchmarking.

---

## 🛠️ System Architecture & Workflow

1.  **Exploratory Data Analysis & Cloud Training**: The `Heart_Disease_Training_Workflow.ipynb` notebook processes the Cleveland dataset, evaluates feature engineering steps, and builds baseline classifiers before exporting the optimal Deep Learning ANN model.
2.  **Modular Pipeline Engine**: The `heart_model.py` serves as the engine, maintaining data-cleaning consistency between the training workspace and production features.
3.  **Interactive User Experience**: The user provides 13 standardized clinical metrics (e.g., age, cholesterol, chest pain type, max heart rate) via the Streamlit frontend.
4.  **Instant Inference & Logging**: The pipeline processes the clinical vector, pushes it to the saved model weights, outputs a definitive risk assessment, and optionally commits the execution record straight to the local `prediction_history.db`.

---

## 🚀 Getting Started

To set up the application environment and launch the system locally, follow these steps:

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You can install all necessary dependencies using the following command:
```bash
pip install streamlit tensorflow pandas numpy scikit-learn
2. Running the Application
Execute the Streamlit dashboard by running:

Bash
streamlit run app.py
📊 Core Features Implemented
Dual Inference Triggers: Provides options for direct screening or institutional logging ("Predict Risk" vs. "Predict and Save to History").

Persistent Historical Audit: An integrated database tracking module enabling clinicians to review past patient records seamlessly inside the app.

Deep Learning Prediction: Employs an optimized Feedforward Artificial Neural Network (ANN) to analyze overlapping diagnostic criteria with precision.

Data Consistency: Embedded preprocessing pipelines ensure zero data leakage and guarantee that real-time runtime inputs match the exact mathematical scaling of the original training set.

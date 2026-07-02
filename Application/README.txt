HEART DISEASE PREDICTION USING DEEP LEARNING
Final Application Package

This folder contains the final application files for the project
"Heart Disease Prediction Using Deep Learning".

PROJECT PURPOSE

This is an academic prototype that uses the Cleveland heart disease dataset
to predict one of three risk classes:

- 0 -> Low Risk
- 1, 2 -> Medium Risk
- 3, 4 -> High Risk

The application is for demonstration and final-project evaluation only. It is
not a medical diagnosis tool and should not be used as medical advice.

REQUIREMENT ALIGNMENT

- Multi-class classification: Low Risk, Medium Risk, High Risk.
- Patient health input form with validation.
- Missing-value handling and normalization in the training pipeline.
- Feedforward TensorFlow/Keras ANN using Dense layers with Dropout and L2
  regularization.
- Logistic Regression baseline model.
- Random Forest comparison model.
- Saved trained model artifact loaded by the app; the app does not retrain on
  startup.
- Prediction probabilities, charts, model metrics, confusion matrix, and report
  download.
- Patient prediction history stored in a database file, prediction_history.db.

FILES INCLUDED

1. app.py
   Main Streamlit web application. It loads the trained model, collects
   validated patient inputs, predicts risk, and can save prediction history to
   a local SQLite database.

2. heart_model.py
   Shared model code used by both the training script and the application.
   It contains dataset loading, preprocessing, ANN, Logistic Regression,
   Random Forest, evaluation, and model artifact loading helpers.

3. train_model.py
   One-time training script. Run this file when you need to create or refresh
   the saved model artifact.

4. model/heart_disease_ann.keras
   Saved TensorFlow/Keras ANN model loaded by the Streamlit app for prediction.

5. model/preprocessor.joblib
   Saved preprocessing pipeline used with the ANN model.

6. model/metrics.json
   Saved model comparison and evaluation metrics generated during training.

7. run_streamlit_app.bat
   Windows launcher that installs/checks dependencies and starts the app.

8. final_heart_disease_prediction_colab.ipynb
   Supporting Google Colab notebook used for preprocessing, model training,
   evaluation, and experiment evidence.

9. processed.cleveland.data
   Cleveland heart disease dataset used by the training script and notebook.

10. requirements.txt
   Python libraries required to run the application and notebook.

11. prediction_history.db
   SQLite database file created by the app when saved predictions are
   generated.

HOW TO RUN THE APPLICATION

Option 1: Use the Windows launcher

1. Open this Application folder.
2. Double-click run_streamlit_app.bat.
3. The app will open in your browser after Streamlit starts.

Option 2: Run from Command Prompt or PowerShell

1. Open a terminal in this Application folder.
2. Use Python 3.12, then install dependencies:

   python -m pip install -r requirements.txt

3. Train the model once if the model artifact is missing:

   python train_model.py

4. Start the Streamlit app:

   python -m streamlit run app.py

If python does not work on your system and Python 3.12 is installed, try:

   py -3.12 -m pip install -r requirements.txt
   py -3.12 train_model.py
   py -3.12 -m streamlit run app.py

NOTES

- The Streamlit app does not train the model at startup. It loads the saved
  model artifact from the model folder.
- Run train_model.py only when the model file is missing or when you need to
  retrain after changing model code or data.
- Keep app.py, heart_model.py, train_model.py, and processed.cleveland.data in
  the same folder.
- Prediction history is saved to prediction_history.db in a SQLite table named
  predictions.
- The teacher's document lists MySQL/PostgreSQL as database options. This
  package uses SQLite because it is a local database file and does not require
  external credentials. It can be migrated to MySQL/PostgreSQL later without
  changing the machine-learning model.
- Patient ID is displayed only for dataset preview and is not used for model
  training.

GOOGLE COLAB KERAS ANN WORKFLOW

Use this workflow when you want to train the teacher-required TensorFlow/Keras
ANN with Dropout and L2 regularization.

1. Open final_heart_disease_prediction_colab.ipynb in Google Colab.
2. Upload these files when the notebook asks:

   heart_model.py
   processed.cleveland.data

3. Run all notebook cells from top to bottom.
4. The notebook will train:

   - Feedforward ANN with Dropout and L2 regularization
   - Logistic Regression baseline
   - Random Forest comparison model

5. The notebook will create and download:

   model_artifacts_for_streamlit.zip

6. Unzip the downloaded file.
7. Replace this project's local model folder with the downloaded model folder.
8. Start the Streamlit app.

Important local note:

- The final application expects the TensorFlow/Keras ANN artifact
  heart_disease_ann.keras and preprocessor.joblib. This keeps the notebook,
  training script, and app aligned to the same model pipeline.

Prepared for final project submission.
Group ID: F25PROJECT7FF99
Supervisor: Laraib Sana

@echo off
cd /d "%~dp0"

echo ======================================
echo Heart Disease Prediction App Launcher
echo ======================================
echo.

if not exist "requirements.txt" (
    echo requirements.txt was not found in this Application folder.
    echo Please extract the complete FinalDeliverables_F25PROJECT7FF99.zip first,
    echo then run this file from inside the extracted Application folder.
    pause
    exit /b 1
)

py -3.12 --version >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3.12"
) else (
    python --version 2>nul | findstr "3.12" >nul
    if %errorlevel%==0 (
        set "PY_CMD=python"
    ) else (
        echo Python 3.12 is required for TensorFlow/Keras compatibility.
        echo Install Python 3.12 first, then run this launcher again.
        pause
        exit /b 1
    )
)

echo Installing/checking required packages...
%PY_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package installation failed.
    echo Make sure internet is available and pip is working.
    pause
    exit /b 1
)

echo.
if not exist "model\heart_disease_ann.keras" (
    echo TensorFlow/Keras ANN model not found. Training model once...
    %PY_CMD% train_model.py
    if errorlevel 1 (
        echo.
        echo Model training failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit app...
%PY_CMD% -m streamlit run app.py
pause

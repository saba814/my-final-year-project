import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'processed.cleveland.data'
DB_FILE = BASE_DIR / 'prediction_history.db'
MODEL_DIR = BASE_DIR / 'model'
METRICS_FILE = MODEL_DIR / 'metrics.json'
KERAS_MODEL_FILE = MODEL_DIR / 'heart_disease_ann.keras'
PREPROCESSOR_FILE = MODEL_DIR / 'preprocessor.joblib'

COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
FEATURES = COLUMNS[:-1]
DISPLAY_COLUMNS = ['PtID'] + COLUMNS + ['risk_class']
RISK_LABELS = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}


class KerasANNPredictor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict_proba(self, input_df: pd.DataFrame):
        transformed = self.preprocessor.transform(input_df[FEATURES])
        return self.model.predict(transformed, verbose=0)


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


def map_risk_class(value: int) -> int:
    if value == 0:
        return 0
    if value in [1, 2]:
        return 1
    return 2


def load_dataset(add_patient_id: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, names=COLUMNS)
    df = df.replace('?', np.nan)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['target'] = df['target'].fillna(0).astype(int)
    df['risk_class'] = df['target'].apply(map_risk_class)

    if add_patient_id and 'PtID' not in df.columns:
        df.insert(0, 'PtID', range(1, len(df) + 1))

    return df


def build_preprocessor(with_scaling: bool = True) -> ColumnTransformer:
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if with_scaling:
        steps.append(('scaler', StandardScaler()))
    return ColumnTransformer(
        transformers=[('num', Pipeline(steps=steps), FEATURES)]
    )


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor(with_scaling=True)),
        ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)),
    ])


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor(with_scaling=False)),
        ('model', RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42,
        )),
    ])


def build_keras_ann(input_dim: int):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
        ),
        layers.Dropout(0.30),
        layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
        ),
        layers.Dropout(0.20),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def evaluate_predictions(y_test, predictions):
    return {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'classification_report_text': classification_report(
            y_test,
            predictions,
            target_names=list(RISK_LABELS.values()),
            zero_division=0,
        ),
        'classification_report': classification_report(
            y_test,
            predictions,
            target_names=list(RISK_LABELS.values()),
            output_dict=True,
            zero_division=0,
        ),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
    }


def train_keras_ann(X_train, X_test, y_train, y_test):
    import tensorflow as tf

    preprocessor = build_preprocessor(with_scaling=True)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    model = build_keras_ann(X_train_processed.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train_processed,
        y_train,
        validation_split=0.15,
        epochs=250,
        batch_size=16,
        callbacks=callbacks,
        verbose=0,
    )
    probabilities = model.predict(X_test_processed, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    result = evaluate_predictions(y_test, predictions)
    result.update({
        'model_type': 'TensorFlow/Keras Feedforward ANN',
        'dropout': [0.30, 0.20],
        'regularization': 'L2 kernel regularization, lambda=0.001',
        'epochs_trained': len(history.history['loss']),
    })
    return model, preprocessor, result


def train_sklearn_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    result = evaluate_predictions(y_test, predictions)
    result['model_type'] = name
    return pipeline, result


def train_and_evaluate():
    df = load_dataset()
    X = df[FEATURES]
    y = df['risk_class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if not tensorflow_available():
        raise RuntimeError(
            'TensorFlow/Keras is required for final ANN training. '
            'Use Python 3.12, install requirements.txt, then rerun train_model.py.'
        )

    keras_model, preprocessor, keras_metrics = train_keras_ann(
        X_train, X_test, y_train, y_test
    )

    logreg_pipeline, logreg_metrics = train_sklearn_model(
        'Logistic Regression baseline',
        build_logistic_regression_pipeline(),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    rf_pipeline, rf_metrics = train_sklearn_model(
        'Random Forest comparison model',
        build_random_forest_pipeline(),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    comparison_models = [
        ('ANN', keras_model, preprocessor, keras_metrics),
        ('Logistic Regression', logreg_pipeline, None, logreg_metrics),
        ('Random Forest', rf_pipeline, None, rf_metrics),
    ]

    comparison = [
        {
            'model': label,
            'model_type': metrics['model_type'],
            'accuracy': metrics['accuracy'],
        }
        for label, _, _, metrics in comparison_models
    ]
    selected_label, selected_model, selected_preprocessor, selected_metrics = comparison_models[0]

    metrics = {
        **selected_metrics,
        'selected_model': selected_label,
        'comparison': comparison,
        'tensorflow_available': True,
        'test_size': 0.2,
        'random_state': 42,
        'features': FEATURES,
        'risk_labels': RISK_LABELS,
        'training_note': (
            'TensorFlow/Keras ANN with Dropout and L2 regularization was trained once. '
            'Logistic Regression and Random Forest are comparison models only; '
            'the Streamlit app loads the saved ANN artifact for prediction.'
        ),
    }
    return selected_label, selected_model, selected_preprocessor, metrics


def save_model_artifacts():
    MODEL_DIR.mkdir(exist_ok=True)
    selected_label, model, preprocessor, metrics = train_and_evaluate()

    model.save(KERAS_MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    metrics['artifact_type'] = 'keras'
    metrics['model_file'] = str(KERAS_MODEL_FILE)
    metrics['preprocessor_file'] = str(PREPROCESSOR_FILE)

    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def load_trained_model():
    metrics = load_metrics()
    if metrics and metrics.get('artifact_type') != 'keras':
        raise RuntimeError(
            'The saved metrics do not describe the final TensorFlow/Keras ANN artifact. '
            'Use Python 3.12 and rerun train_model.py.'
        )
    if not KERAS_MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError('Keras model artifacts are missing. Run `python train_model.py`.')

    import tensorflow as tf

    return KerasANNPredictor(
        joblib.load(PREPROCESSOR_FILE),
        tf.keras.models.load_model(KERAS_MODEL_FILE),
    )


def load_metrics():
    if not METRICS_FILE.exists():
        return None
    return json.loads(METRICS_FILE.read_text(encoding='utf-8-sig'))

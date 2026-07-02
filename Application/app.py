from datetime import datetime
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from heart_model import (
    DATA_FILE,
    DB_FILE,
    DISPLAY_COLUMNS,
    FEATURES,
    KERAS_MODEL_FILE,
    PREPROCESSOR_FILE,
    RISK_LABELS,
    load_dataset,
    load_metrics,
    load_trained_model,
)

st.set_page_config(page_title='Heart Disease Prediction App', page_icon='🫀', layout='wide')

PREDICTIONS_TABLE = 'predictions'

DEFAULTS = {
    'age': 54,
    'sex': 1,
    'cp': 4,
    'trestbps': 130,
    'chol': 246,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.2,
    'slope': 1,
    'ca': 0,
    'thal': 3,
}

FIELD_CONFIG = {
    'age': {
        'label': 'Age',
        'help': 'Age in years.',
        'min': 18,
        'max': 100,
        'step': 1,
    },
    'sex': {
        'label': 'Sex',
        'help': 'Patient sex recorded in the Cleveland dataset.',
        'options': {0: 'Female', 1: 'Male'},
    },
    'cp': {
        'label': 'Chest Pain Type',
        'help': 'Cleveland dataset chest pain type category.',
        'options': {
            1: '1 - Typical angina',
            2: '2 - Atypical angina',
            3: '3 - Non-anginal pain',
            4: '4 - Asymptomatic',
        },
    },
    'trestbps': {
        'label': 'Resting Blood Pressure',
        'help': 'Resting blood pressure in mm Hg.',
        'min': 80,
        'max': 220,
        'step': 1,
    },
    'chol': {
        'label': 'Serum Cholesterol',
        'help': 'Serum cholesterol in mg/dl.',
        'min': 100,
        'max': 600,
        'step': 1,
    },
    'fbs': {
        'label': 'Fasting Blood Sugar',
        'help': 'Whether fasting blood sugar is greater than 120 mg/dl.',
        'options': {0: '0 - <= 120 mg/dl', 1: '1 - > 120 mg/dl'},
    },
    'restecg': {
        'label': 'Resting ECG',
        'help': 'Resting electrocardiographic result.',
        'options': {
            0: '0 - Normal',
            1: '1 - ST-T wave abnormality',
            2: '2 - Left ventricular hypertrophy',
        },
    },
    'thalach': {
        'label': 'Maximum Heart Rate',
        'help': 'Maximum heart rate achieved.',
        'min': 60,
        'max': 220,
        'step': 1,
    },
    'exang': {
        'label': 'Exercise Induced Angina',
        'help': 'Whether exercise induced angina was observed.',
        'options': {0: 'No', 1: 'Yes'},
    },
    'oldpeak': {
        'label': 'ST Depression',
        'help': 'ST depression induced by exercise relative to rest.',
        'min': 0.0,
        'max': 7.0,
        'step': 0.1,
    },
    'slope': {
        'label': 'ST Segment Slope',
        'help': 'Slope of the peak exercise ST segment.',
        'options': {1: '1 - Upsloping', 2: '2 - Flat', 3: '3 - Downsloping'},
    },
    'ca': {
        'label': 'Major Vessels',
        'help': 'Number of major vessels colored by fluoroscopy.',
        'options': {0: '0', 1: '1', 2: '2', 3: '3'},
    },
    'thal': {
        'label': 'Thalassemia',
        'help': 'Thalassemia category from the Cleveland dataset.',
        'options': {3: '3 - Normal', 6: '6 - Fixed defect', 7: '7 - Reversible defect'},
    },
}

@st.cache_data(show_spinner=False)
def load_data():
    return load_dataset(add_patient_id=True)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_trained_model()


@st.cache_data(show_spinner=False)
def get_model_metrics():
    return load_metrics()


def save_prediction(patient_info: dict, values: dict, pred_label: str, probs: list[float]):
    init_db()
    record = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        **patient_info,
        **values,
        'prediction': pred_label,
        'low_risk_probability': round(float(probs[0]), 4),
        'medium_risk_probability': round(float(probs[1]), 4),
        'high_risk_probability': round(float(probs[2]), 4),
    }
    columns = ', '.join(record.keys())
    placeholders = ', '.join(['?'] * len(record))
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f'INSERT INTO {PREDICTIONS_TABLE} ({columns}) VALUES ({placeholders})',
            list(record.values()),
        )


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_record_id TEXT,
                patient_name TEXT,
                age INTEGER NOT NULL,
                sex INTEGER NOT NULL,
                cp INTEGER NOT NULL,
                trestbps INTEGER NOT NULL,
                chol INTEGER NOT NULL,
                fbs INTEGER NOT NULL,
                restecg INTEGER NOT NULL,
                thalach INTEGER NOT NULL,
                exang INTEGER NOT NULL,
                oldpeak REAL NOT NULL,
                slope INTEGER NOT NULL,
                ca INTEGER NOT NULL,
                thal INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                low_risk_probability REAL NOT NULL,
                medium_risk_probability REAL NOT NULL,
                high_risk_probability REAL NOT NULL
            )
            """
        )
        existing_columns = {
            row[1] for row in conn.execute(f'PRAGMA table_info({PREDICTIONS_TABLE})')
        }
        if 'patient_record_id' not in existing_columns:
            conn.execute(f'ALTER TABLE {PREDICTIONS_TABLE} ADD COLUMN patient_record_id TEXT')
        if 'patient_name' not in existing_columns:
            conn.execute(f'ALTER TABLE {PREDICTIONS_TABLE} ADD COLUMN patient_name TEXT')


def load_prediction_history(limit: int = 20):
    init_db()
    query = f"""
        SELECT
            id,
            timestamp,
            patient_record_id,
            patient_name,
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
            prediction,
            low_risk_probability,
            medium_risk_probability,
            high_risk_probability
        FROM {PREDICTIONS_TABLE}
        ORDER BY id DESC
        LIMIT ?
    """
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(query, conn, params=(limit,))


def render_input(field: str):
    config = FIELD_CONFIG[field]
    default = DEFAULTS[field]
    if 'options' in config:
        options = list(config['options'].keys())
        default_index = options.index(default)
        return st.selectbox(
            config['label'],
            options=options,
            index=default_index,
            format_func=lambda value: config['options'][value],
            help=config['help'],
        )

    if isinstance(default, float):
        return st.number_input(
            config['label'],
            min_value=float(config['min']),
            max_value=float(config['max']),
            value=float(default),
            step=float(config['step']),
            format='%.1f',
            help=config['help'],
        )

    return st.number_input(
        config['label'],
        min_value=int(config['min']),
        max_value=int(config['max']),
        value=int(default),
        step=int(config['step']),
        format='%d',
        help=config['help'],
    )


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style='padding:16px;border-radius:14px;background:#f6f8fb;border:1px solid #dfe6ee;'>
            <div style='font-size:14px;color:#4c5563;'>{label}</div>
            <div style='font-size:28px;font-weight:700;color:#0f172a;'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_prediction_report(patient_info: dict, values: dict, pred_label: str, probs: np.ndarray) -> str:
    lines = [
        'Heart Disease Prediction Report',
        f'Generated At: {datetime.now().isoformat(timespec="seconds")}',
        '',
        'Patient Identity:',
        f'- Patient/Record ID: {patient_info["patient_record_id"] or "Not provided"}',
        f'- Patient Name: {patient_info["patient_name"] or "Not provided"}',
        '',
        'Patient Inputs:',
    ]
    for field in FEATURES:
        lines.append(f'- {FIELD_CONFIG[field]["label"]}: {values[field]}')
    lines.extend([
        '',
        f'Predicted Risk: {pred_label}',
        '',
        'Prediction Probabilities:',
        f'- Low Risk: {probs[0]:.4f}',
        f'- Medium Risk: {probs[1]:.4f}',
        f'- High Risk: {probs[2]:.4f}',
        '',
        'Disclaimer: This report is for academic demonstration only and is not medical advice.',
    ])
    return '\n'.join(lines)


st.title('🫀 Heart Disease Prediction Application')
st.caption('Academic prototype with a simple interactive interface for final project submission')

if not DATA_FILE.exists():
    st.error('Dataset file `processed.cleveland.data` not found in the same folder as app.py.')
    st.stop()

metrics = get_model_metrics()
has_keras_model = KERAS_MODEL_FILE.exists() and PREPROCESSOR_FILE.exists()

if not has_keras_model:
    st.error(
        'TensorFlow/Keras ANN artifact not found. Run `python train_model.py` once '
        'with Python 3.12 before starting the app.'
    )
    st.stop()

init_db()
clf = get_model()

with st.sidebar:
    st.header('About this app')
    st.write(
        'This interface uses the Cleveland heart disease dataset and a neural-network-based classifier '
        'to predict Low, Medium, or High heart disease risk.'
    )
    st.info('Academic demonstration only. This app is not a medical diagnosis tool.')
    st.write('Use the Predict tab to enter patient values and generate a result.')

home_tab, predict_tab, history_tab, dataset_tab = st.tabs([
    'Overview',
    'Predict Risk',
    'Prediction History',
    'Dataset & Model',
])

with home_tab:
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card('Application Type', 'Streamlit Web App')
    with col2:
        metric_card('Prediction Classes', 'Low / Medium / High')
    with col3:
        accuracy = metrics['accuracy'] if metrics else 0
        metric_card('Model Accuracy', f'{accuracy:.2%}')

    if metrics:
        st.subheader('Model Comparison')
        comparison_df = pd.DataFrame(metrics.get('comparison', []))
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.bar_chart(comparison_df.set_index('model')['accuracy'])
        st.caption(metrics.get('training_note', ''))

    st.subheader('Workflow')
    st.markdown(
        '1. Enter patient clinical attributes.  '\
        '\n2. The app preprocesses the values using the same pipeline used during training.  '\
        '\n3. The neural network predicts the risk level.  '\
        '\n4. Prediction probabilities and a short interpretation are shown.  '\
        '\n5. The result may also be saved into prediction history.'
    )

with predict_tab:
    st.subheader('Patient Input Form')
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        patient_record_id = st.text_input(
            'Patient/Record ID',
            value='',
            placeholder='Example: P-001 or MR-2026-001',
            help='Used only to recognize saved history. It is not used by the model.',
        ).strip()
    with id_col2:
        patient_name = st.text_input(
            'Patient Name',
            value='',
            placeholder='Example: Ali Khan',
            help='Used only to recognize saved history. It is not used by the model.',
        ).strip()

    patient_info = {
        'patient_record_id': patient_record_id,
        'patient_name': patient_name,
    }

    cols = st.columns(3)
    values = {}
    for idx, field in enumerate(FEATURES):
        with cols[idx % 3]:
            values[field] = render_input(field)

    colp1, colp2 = st.columns([1, 1])
    predict_clicked = colp1.button('🫀 Predict Risk', use_container_width=True)
    save_clicked = colp2.button('🫀 Predict and Save to History', use_container_width=True)

    if predict_clicked or save_clicked:
        input_df = pd.DataFrame([values])
        probs = clf.predict_proba(input_df)[0]
        pred = int(np.argmax(probs))
        pred_label = RISK_LABELS[pred]

        st.success(f'🫀 Predicted Result: **{pred_label}**')
        st.write('### Prediction Probabilities')
        prob_df = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Probability': probs
        })
        st.bar_chart(prob_df.set_index('Risk Level'))

        if pred == 0:
            st.info('Interpretation: The current input pattern indicates comparatively lower risk.')
        elif pred == 1:
            st.warning('Interpretation: The current input pattern indicates a moderate level of risk and should be reviewed carefully.')
        else:
            st.error('Interpretation: The current input pattern indicates high risk and needs serious medical attention.')
        st.caption('This result is for academic demonstration only and should not be used as medical advice.')
        st.download_button(
            'Download Prediction Report',
            data=build_prediction_report(patient_info, values, pred_label, probs),
            file_name='heart_disease_prediction_report.txt',
            mime='text/plain',
            use_container_width=True,
        )

        if save_clicked:
            save_prediction(patient_info, values, pred_label, probs)
            st.success('Prediction saved to prediction_history.db')

with history_tab:
    st.subheader('Prediction History')
    st.write('Saved prediction records are loaded from the local SQLite database. Patient/Record ID and Patient Name are stored only for recognizing history records; they are not used for model training or prediction.')
    history_df = load_prediction_history(limit=100)
    if history_df.empty:
        st.info('No saved predictions found yet. Use the Predict Risk tab and select "Predict and Save to History".')
    else:
        metric_card('Saved Records', str(len(history_df)))
        display_history_df = history_df.rename(columns={
            'patient_record_id': 'Patient/Record ID',
            'patient_name': 'Patient Name',
            'timestamp': 'Saved At',
            'prediction': 'Predicted Risk',
            'low_risk_probability': 'Low Risk Probability',
            'medium_risk_probability': 'Medium Risk Probability',
            'high_risk_probability': 'High Risk Probability',
        }).fillna('')
        priority_columns = [
            'id',
            'Saved At',
            'Patient/Record ID',
            'Patient Name',
            'Predicted Risk',
            'Low Risk Probability',
            'Medium Risk Probability',
            'High Risk Probability',
        ]
        ordered_columns = priority_columns + [
            column for column in display_history_df.columns if column not in priority_columns
        ]
        st.dataframe(display_history_df[ordered_columns], use_container_width=True, hide_index=True)
        trend_df = history_df['prediction'].value_counts().rename_axis('Risk Level').reset_index(name='Count')
        st.subheader('Saved Prediction Trend')
        st.bar_chart(trend_df.set_index('Risk Level'))
    st.write('Prediction database file:', DB_FILE.name)

with dataset_tab:
    df = load_data()
    st.subheader('Dataset Preview')
    st.dataframe(df[DISPLAY_COLUMNS].head(10), use_container_width=True)
    st.subheader('Dataset Risk Distribution')
    risk_counts = df['risk_class'].map(RISK_LABELS).value_counts().reindex(RISK_LABELS.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=ax)
    ax.set_xlabel('Risk Category')
    ax.set_ylabel('Records')
    ax.set_title('Grouped Risk Distribution')
    st.pyplot(fig)
    st.subheader('Model Notes')
    st.write('PtID in the dataset preview is only a generated row number for display. Saved prediction history uses Patient/Record ID and Patient Name entered in the prediction form. These identity fields are not used during model training or prediction.')
    st.write('Target values from the original Cleveland dataset were grouped into three classes for academic risk prediction:')
    st.write('- 0 -> Low Risk')
    st.write('- 1, 2 -> Medium Risk')
    st.write('- 3, 4 -> High Risk')
    st.subheader('Classification Report')
    if metrics:
        st.code(metrics['classification_report_text'])
        cm = metrics.get('confusion_matrix')
        if cm:
            st.subheader('Confusion Matrix')
            cm_df = pd.DataFrame(cm, index=RISK_LABELS.values(), columns=RISK_LABELS.values())
            st.dataframe(cm_df, use_container_width=True)
    else:
        st.info('No metrics file found. Run `python train_model.py` to regenerate model metrics.')

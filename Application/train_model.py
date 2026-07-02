from heart_model import METRICS_FILE, save_model_artifacts


def main():
    metrics = save_model_artifacts()
    print(f"Selected model: {metrics['selected_model']}")
    print(f"Model artifact type: {metrics['artifact_type']}")
    print(f"Model saved to: {metrics['model_file']}")
    print(f'Metrics saved to: {METRICS_FILE}')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print('\nModel comparison:')
    for row in metrics['comparison']:
        print(f"- {row['model']}: {row['accuracy']:.4f} ({row['model_type']})")
    print('\nClassification report:')
    print(metrics['classification_report_text'])


if __name__ == '__main__':
    main()

from pycaret.classification import setup, create_model, tune_model, predict_model, pull, models, get_config
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def forecast(data: pd.DataFrame, test: pd.DataFrame, target: str, fold=10, n_jobs=-1, class_weight=None , n_iter=100, model_used='rf'):
    "forecast test data"
    # Setup and cross-validation on train
    setup(data, target=target, fold=fold, n_jobs=n_jobs, verbose=False, session_id=42, fix_imbalance=False, ignore_features=['Date','Close','Volume'])
    
    # Create a temporary model to get the class mapping
    temp_model = create_model(model_used)
    # Retrieve the classes directly from the model
    class_names = temp_model.classes_

    # Rebuild the weights dictionary with numeric indices
    if class_weight is not None:
        class_weight_numeric = {i: class_weight.get(name, 1) for i, name in enumerate(class_names)}
    else:
        class_weight_numeric = None

    # Now create the final model with the correct weights
    model = create_model(model_used, class_weight=class_weight_numeric)
    tuned_model = tune_model(model, n_iter=n_iter, optimize='F1', early_stopping=True)

    # Cross-validation metrics (train)
    cv_metrics = pull()

    # Predictions on train (internal evaluation)
    train_preds = predict_model(tuned_model, data=data)
    train_acc = accuracy_score(train_preds[target], train_preds['prediction_label'])
    train_report = classification_report(train_preds[target], train_preds['prediction_label'], zero_division=0, output_dict=True)

    # Predictions on test
    preds = predict_model(tuned_model, data=test)
    test_acc = accuracy_score(preds[target], preds['prediction_label'])
    test_report = classification_report(preds[target], preds['prediction_label'], zero_division=0, output_dict=True)

    metrics = {
        'cv_metrics': cv_metrics,
        'train_accuracy': train_acc,
        'train_report': train_report,
        'test_accuracy': test_acc,
        'test_report': test_report,
        'model': tuned_model
    }
    return preds, metrics


def print_metrics(metrics):
    print("\n--- Cross-validation metrics (train) ---")
    if 'cv_metrics' in metrics and metrics['cv_metrics'] is not None:
        print(metrics['cv_metrics'])
    else:
        print("No cross-validation metrics available.")
    print("\n--- Train set classification report ---")
    for label, scores in metrics.get('train_report', {}).items():
        print(f"{label}: {scores}")
    if 'train_accuracy' in metrics:
        print(f"Train accuracy: {metrics['train_accuracy']:.4f}")

    print("\n--- Test set classification report ---")
    for label, scores in metrics.get('test_report', {}).items():
        print(f"{label}: {scores}")
    if 'test_accuracy' in metrics:
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")




















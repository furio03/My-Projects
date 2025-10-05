import pandas as pd
from back_test import back_test_one_symbol
from label import label_data
from indicators import calculate_indicators, add_kmeans, find_best_k
from data_manipulation import shift_df
from forecasting import forecast, print_metrics
from sklearn.metrics import precision_score
from optimize_classification import optimize
import numpy as np
import joblib

df_full = pd.read_csv('AAPL_historical_data_1d.csv', skiprows=[1]) # Use skiprows to header issues
length_full = len(df_full)
length_test = 365

df_train = df_full.iloc[:length_full - length_test].reset_index(drop=True)
df_test = df_full.iloc[length_full - length_test:].reset_index(drop=True)

thresholds = [0.2, 0.3, 0.4]

best_precision = -1
best_shift = None
best_threshold = None
best_preds = None
best_metrics = None

shifts = [1]

for threshold in thresholds:
    df_train_thr = label_data(df_train.copy(), threshold=threshold)
    df_test_thr = label_data(df_test.copy(), threshold=threshold)
    df_train_thr = calculate_indicators(df_train_thr)
    df_test_thr = calculate_indicators(df_test_thr)

    # Find the best k on the train set
    best_k = find_best_k(df_train_thr)
    
    df_train_thr['groups'], kmeans_model, _ = add_kmeans(df_train_thr, k=best_k)
    df_test_thr['groups'] = add_kmeans(df_test_thr, k=best_k, model=kmeans_model)

    df_train_inverted = df_train_thr.iloc[::-1].reset_index(drop=True)
    df_test_inverted = df_test_thr.iloc[::-1].reset_index(drop=True)

    feature_cols = [col for col in df_train_thr.columns if col != 'action']

    for shift in shifts:  
        df_train_shifted = shift_df(shift, df_train_inverted)
        df_test_shifted = shift_df(shift, df_test_inverted)
        df_train_shifted = df_train_shifted.dropna(subset=feature_cols, how='any')
        df_test_shifted = df_test_shifted.dropna(subset=feature_cols, how='any')

        labels = list(df_train_shifted['action'].unique())
        weights = [1 if l in ['buy', 'sell'] else 1 for l in labels]
        class_weight = dict(zip(labels, weights))

        preds, metrics = forecast(
            df_train_shifted,
            df_test_shifted,
            target='action',
            fold=10,
            n_jobs=-1,
            class_weight=class_weight,
            n_iter=150,
            model_used='xgboost'
        )

        test_report = metrics['test_report']
        precision = test_report['macro avg']['precision']

        if precision > best_precision:
            best_precision = precision
            best_shift = shift
            best_threshold = threshold
            best_preds = preds
            best_metrics = metrics

cols = [c for c in ['action', 'prediction_label', 'Close', 'Date', 'prediction_score'] if c in best_preds.columns]
pred_db = best_preds[cols].copy()
print(pred_db.head(50))

print_metrics(best_metrics)

df_train_labeled = label_data(df_train.copy(), threshold=best_threshold)
df_test_labeled = label_data(df_test.copy(), threshold=best_threshold)

print(f"\nTrain action value counts (best threshold={best_threshold}):")
print(df_train_labeled['action'].value_counts())
print(f"\nTest action value counts (best threshold={best_threshold}):")
print(df_test_labeled['action'].value_counts())
print('')
print(f"\nBest threshold: {best_threshold} ")
print(f"\nBest shift: {best_shift} ")

performance = back_test_one_symbol(pred_db, position_value=1000.0)
print("\nPerformance:")
print(performance)
performance.to_csv('performance.csv', index=False)


# Save the best model, shift and threshold found
joblib.dump({
    'model': best_metrics['model'],
    'shift': best_shift,
    'threshold': best_threshold,
    'best_k': best_k
}, 'best_model.pkl')
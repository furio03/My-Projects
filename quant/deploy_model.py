import pandas as pd
from back_test import back_test_one_symbol
from label import label_data
from indicators import calculate_indicators, add_kmeans
from data_manipulation import shift_df
from forecasting import print_metrics
from optimize_classification import optimize
import numpy as np
import joblib
from pycaret.classification import setup, predict_model
from sklearn.metrics import classification_report, accuracy_score

# Load the best model and parameters
loaded_model=joblib.load('best_model.pkl')
best_shift = loaded_model['shift']
best_threshold = loaded_model['threshold']
best_k = loaded_model['best_k']
model = loaded_model['model']

# Load only the test set

df_full = pd.read_csv('AAPL_historical_data_1d.csv', skiprows=[1]) # Use skiprows to avoid header issues
length_full = len(df_full)
length_test = 365
df_test = df_full.iloc[length_full - length_test:].reset_index(drop=True)

# Prepare test data using the best parameters

df_test_thr = label_data(df_test.copy(), threshold=best_threshold)
df_test_thr = calculate_indicators(df_test_thr)

# Apply kmeans with best_k
_, kmeans_model, _ = add_kmeans(df_test_thr, k=best_k)
df_test_thr['groups'] = add_kmeans(df_test_thr, k=best_k, model=kmeans_model)

df_test_inverted = df_test_thr.iloc[::-1].reset_index(drop=True)
feature_cols = [col for col in df_test_thr.columns if col != 'action']

df_test_shifted = shift_df(best_shift, df_test_inverted)
df_test_shifted = df_test_shifted.dropna(subset=feature_cols, how='any')

# PyCaret setup (test only)
setup(
    data=df_test_shifted,
    target='action',
    fold=10,
    n_jobs=-1,
    verbose=False,
    session_id=42,
    fix_imbalance=False,
    ignore_features=['Date','Close','Volume']
)

# Predictions
preds = predict_model(model, data=df_test_shifted)

test_acc = accuracy_score(df_test_shifted['action'], preds['prediction_label'])
test_report = classification_report(df_test_shifted['action'], preds['prediction_label'], zero_division=0, output_dict=True)
metrics = {
    'test_accuracy': test_acc,
    'test_report': test_report,
    'model': model
}

cols = [c for c in ['action', 'prediction_label', 'Close', 'Date', 'prediction_score'] if c in preds.columns]
pred_db = preds[cols].copy()
print(pred_db.head(50))

print_metrics(metrics)

performance = back_test_one_symbol(pred_db, position_value=1000.0)
print("\nPerformance:")
print(performance)
performance.to_csv('performance.csv', index=False)

#Optimization step for result comparability
pred_db_optimized = optimize(pred_db, threshold=best_threshold)
performance_optimized = back_test_one_symbol(pred_db_optimized, position_value=1000.0)
print("\nOptimized performance:")
print(performance_optimized)
performance_optimized.to_csv('performance_optimized.csv', index=False)




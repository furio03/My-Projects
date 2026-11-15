"""Time-ordered model selection with all learned features inside each fold."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    from .indicators import add_kmeans
except ImportError:
    from indicators import add_kmeans


class KMeansFeatures(TransformerMixin, BaseEstimator):
    """Append a cluster feature learned from the current training fold only."""

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        _, self.model_, _ = add_kmeans(X, self.n_clusters)
        return self

    def transform(self, X):
        result = X.copy()
        result['groups'] = add_kmeans(X, self.n_clusters, model=self.model_)
        return result.astype(float)


def temporal_folds(data, labels, folds):
    """Leave one session between training targets and validation observations."""
    splits = list(TimeSeriesSplit(n_splits=folds, gap=1).split(data))
    # XGBoost requires every encoded class in each training partition.
    classes = np.unique(labels)
    splits = [(train, valid) for train, valid in splits
              if np.array_equal(np.unique(labels[train]), classes)]
    if len(splits) < 2:
        raise ValueError('Need at least two temporal folds containing all training classes.')
    return splits


def predict(model, data):
    """Use the fitted pipeline without learning anything from prediction data."""
    result = data.copy()
    features = data[model['features']]
    encoded = model['pipeline'].predict(features).astype(int)
    result['prediction_label'] = model['labels'].inverse_transform(encoded)
    result['prediction_score'] = model['pipeline'].predict_proba(features).max(axis=1)
    return result


def evaluate(predictions, target='action'):
    known = predictions.dropna(subset=[target])
    if known.empty:
        raise ValueError('No known targets are available for evaluation.')
    return {
        'test_accuracy': accuracy_score(known[target], known.prediction_label),
        'test_report': classification_report(
            known[target], known.prediction_label, zero_division=0, output_dict=True,
        ),
    }


def forecast(data, test, target='action', fold=10, n_jobs=-1,
             class_weight=None, n_iter=100, model_used='xgboost'):
    data = data.sort_values('Date').dropna(subset=[target]).copy()
    features = data.columns.difference([target, 'Date', 'Close', 'Volume']).tolist()
    X = data[features].astype(float)
    labels = LabelEncoder().fit(data[target])
    y = labels.transform(data[target])
    if model_used == 'xgboost':
        from xgboost import XGBClassifier
        classifier = XGBClassifier(random_state=42, n_jobs=1)
        parameters = {'classifier__max_depth': [2, 3, 5],
                      'classifier__learning_rate': [0.03, 0.1]}
    elif model_used == 'rf':
        classifier = RandomForestClassifier(random_state=42, n_jobs=1)
        parameters = {'classifier__max_depth': [3, 6, None]}
    else:
        raise ValueError("Model must be 'xgboost' or 'rf'.")
    parameters.update({'classifier__n_estimators': [50, 100, 200],
                       'groups__n_clusters': [2, 3, 4]})
    pipeline = Pipeline([('groups', KMeansFeatures()), ('classifier', classifier)])
    splits = temporal_folds(X, y, fold)
    search = RandomizedSearchCV(
        pipeline, parameters, n_iter=n_iter, cv=splits, n_jobs=n_jobs,
        scoring=make_scorer(f1_score, average='macro', zero_division=0),
        random_state=42, error_score='raise', refit=True,
    )
    sample_weight = np.array([class_weight.get(label, 1) for label in data[target]]) if class_weight else np.ones(len(data))
    search.fit(X, y, classifier__sample_weight=sample_weight)
    model = {'pipeline': search.best_estimator_, 'labels': labels, 'features': features}
    predictions = predict(model, test)
    metrics = evaluate(predictions, target)
    training = evaluate(predict(model, data), target)
    metrics.update({
        'train_accuracy': training['test_accuracy'],
        'train_report': training['test_report'],
        'cv_metrics': pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']],
        'model': model,
    })
    return predictions, metrics


def print_metrics(metrics):
    if 'cv_metrics' in metrics:
        print('\nTemporal cross-validation (macro F1)')
        print(metrics['cv_metrics'].to_string(index=False))
    if 'train_accuracy' in metrics:
        print(f"\nTraining accuracy (in-sample): {metrics['train_accuracy']:.4f}")
    print(f"\nEvaluation accuracy: {metrics['test_accuracy']:.4f}")
    print(pd.DataFrame(metrics['test_report']).T.to_string())

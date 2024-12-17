import logging
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def temporal_train_test_split(X, y, test_size=0.2):
    """
    Splits the data into train and test sets based on time order.
    Ensures that training data precedes testing data.
    """
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, mae, rmse

def get_models():
    """
    Returns a dictionary of models to try.
    Keys are model names, values are model instances.
    """
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(),
        'random_forest': RandomForestRegressor(random_state=42),
        'mlp': MLPRegressor(random_state=42, max_iter=500)
    }
    return models

def get_hyperparameter_grids():
    """
    Returns a dictionary of hyperparameter grids for tuning.
    Keys should match keys in get_models().
    """
    grids = {
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20]
        },
        # Add more grids here for other models as needed
    }
    return grids

def train_and_tune_model(model_name, model, X_train, y_train, cv_splits=3):
    """
    Trains and optionally tunes a given model using TimeSeriesSplit for CV.
    If hyperparameters are available for the model in get_hyperparameter_grids(),
    GridSearchCV or RandomizedSearchCV is used.

    Returns the best trained model.
    """
    grids = get_hyperparameter_grids()
    if model_name in grids:
        logging.info(f"Hyperparameter tuning for {model_name}...")
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # We can choose GridSearchCV or RandomizedSearchCV here.
        # Using GridSearchCV for simplicity.
        grid_search = GridSearchCV(model, grids[model_name], cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        # No hyperparameter tuning, just fit directly
        model.fit(X_train, y_train)
        return model

def quick_cross_validation(model, X, y, cv_splits=3):
    """
    Performs a quick cross-validation using TimeSeriesSplit to validate temporal performance.
    Logs the average MAE and RMSE scores.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    maes = []
    rmses = []
    for train_index, test_index in tscv.split(X):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        model_clone = clone_model(model)
        model_clone.fit(X_train_cv, y_train_cv)
        _, mae, rmse = evaluate_model(model_clone, X_test_cv, y_test_cv)
        maes.append(mae)
        rmses.append(rmse)

    logging.info(f"Cross-validation results: Average MAE={np.mean(maes):.4f}, Average RMSE={np.mean(rmses):.4f}")

def clone_model(model):
    """
    Creates a clone of the given model. For simplicity, re-instantiates a model of the same type and parameters.
    """
    from sklearn.base import clone
    return clone(model)

def train_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate all models. Returns a dictionary of results and the best model found.
    """
    models = get_models()
    results = {}
    best_model = None
    best_rmse = float('inf')

    for name, model in models.items():
        logging.info(f"Training model: {name}")
        tuned_model = train_and_tune_model(name, model, X_train, y_train)
        y_pred, mae, rmse = evaluate_model(tuned_model, X_test, y_test)
        logging.info(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

        results[name] = {
            'model': tuned_model,
            'y_pred': y_pred,
            'mae': mae,
            'rmse': rmse
        }

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = tuned_model

    return results, best_model

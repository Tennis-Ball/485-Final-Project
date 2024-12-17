# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from scipy.stats import ttest_rel, wilcoxon

from data_collection import download_filings, collect_filings, align_data
from data_preprocessing import preprocess_filings
from feature_extraction import extract_features
from model_training import temporal_train_test_split, train_models, evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-9))  # add small epsilon to avoid div by zero

def mape(y_true, y_pred):
    # Mean Absolute Percentage Error, ignoring zero values in y_true
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def plot_actual_vs_predicted(y_test, y_pred, model_name="model", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(range(len(y_test)), y_test, label='Actual Volatility', marker='o')
    plt.plot(range(len(y_pred)), y_pred, label=f'Predicted Volatility ({model_name})', marker='x')
    plt.xlabel('Sample')
    plt.ylabel('Volatility')
    plt.title(f'Actual vs. Predicted Volatility - {model_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'actual_vs_pred_{model_name}.png'))
    plt.close()

def plot_distributions(y_test, y_pred, model_name="model", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_test - y_pred

    # Histogram of actual vs predicted
    plt.figure(figsize=(10,6))
    plt.hist(y_test, alpha=0.5, bins=20, label='Actual', color='blue')
    plt.hist(y_pred, alpha=0.5, bins=20, label='Predicted', color='orange')
    plt.xlabel('Volatility')
    plt.ylabel('Frequency')
    plt.title(f'Volatility Distribution - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'volatility_distribution_{model_name}.png'))
    plt.close()

    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=10, color='red', edgecolor='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution - {model_name}')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'residual_distribution_{model_name}.png'))
    plt.close()

def compare_models_boxplot(results, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    # Collect residuals for each model
    residuals_dict = {}
    for model_name, res in results.items():
        y_pred = res['y_pred']
        y_test = res['y_test']
        residuals = y_test - y_pred
        residuals_dict[model_name] = residuals

    # Create a boxplot
    plt.figure(figsize=(10,6))
    plt.boxplot(residuals_dict.values(), labels=residuals_dict.keys())
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Distribution Across Models')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'residuals_boxplot.png'))
    plt.close()

def statistical_significance_test(results):
    # Compare best two models based on RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
    if len(sorted_results) < 2:
        logging.info("Not enough models for statistical comparison.")
        return

    best_model_name, best_res = sorted_results[0]
    second_model_name, second_res = sorted_results[1]

    best_residuals = best_res['y_test'] - best_res['y_pred']
    second_residuals = second_res['y_test'] - second_res['y_pred']

    # Perform paired t-test
    t_stat, p_value = ttest_rel(best_residuals, second_residuals)
    logging.info(f"Paired t-test between {best_model_name} and {second_model_name}: t={t_stat:.4f}, p={p_value:.4f}")

    # Optional: Wilcoxon signed-rank test
    w_stat, w_p_value = wilcoxon(best_residuals, second_residuals)
    logging.info(f"Wilcoxon test between {best_model_name} and {second_model_name}: W={w_stat}, p={w_p_value:.4f}")

def main():
    # Step 1: Define tickers and download filings
    tickers = [
        "AAPL",
    ]
    download_filings(tickers, num_filings=40)
    
    # Step 2: Collect and align filings with stock data
    filings = collect_filings(tickers)
    filings = align_data(filings, normalize_volatility=False)
    filings = [f for f in filings if f['Volatility'] is not None and f['Text']]
    
    if not filings:
        logging.warning("No filings with relevant text found. Exiting.")
        return

    # Step 3: Preprocess text data
    filings = preprocess_filings(filings)
    
    # Convert to DataFrame
    filings_df = pd.DataFrame(filings)
    if 'Cleaned Text' not in filings_df.columns:
        logging.warning("No cleaned text found after preprocessing. Exiting.")
        return

    # Step 4: Feature extraction
    X, vectorizer, lda_model, lda_vectorizer = extract_features(filings)
    
    # Step 5: Prepare target variable
    y = filings_df['Volatility'].values
    
    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
    
    # Step 7: Train models
    results, best_model = train_models(X_train, y_train, X_test, y_test)

    # Add metrics (R², MAPE) to results and store actual y_test in results
    for model_name, res in results.items():
        y_pred = res['y_pred']
        mae = res['mae']
        rmse = res['rmse']
        r2 = r2_score(y_test, y_pred)
        mape_val = mape(y_test, y_pred)
        results[model_name]['r2'] = r2
        results[model_name]['mape'] = mape_val
        results[model_name]['y_test'] = y_test  # store for plotting
        logging.info(f"{model_name}: R²={r2:.4f}, MAPE={mape_val:.2f}%")

    # Print summary table
    summary_data = []
    for model_name, res in results.items():
        summary_data.append({
            'Model': model_name,
            'MAE': res['mae'],
            'RMSE': res['rmse'],
            'R²': res['r2'],
            'MAPE (%)': res['mape']
        })
    summary_df = pd.DataFrame(summary_data)
    logging.info("Model Performance Summary:")
    logging.info("\n" + summary_df.to_string(index=False))

    # Save summary to CSV
    summary_df.to_csv('model_performance_summary.csv', index=False)

    # Visualizations
    for model_name, res in results.items():
        y_pred = res['y_pred']
        plot_actual_vs_predicted(y_test, y_pred, model_name=model_name)
        plot_distributions(y_test, y_pred, model_name=model_name)

    compare_models_boxplot(results)

    # Statistical significance test between top models
    statistical_significance_test(results)

    logging.info("Evaluation and visualization completed.")

if __name__ == "__main__":
    main()

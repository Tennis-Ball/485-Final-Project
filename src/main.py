# main.py

import pandas as pd
import matplotlib.pyplot as plt

from data_collection import download_filings, collect_filings, align_data
from data_preprocessing import preprocess_filings
from feature_extraction import extract_features
from model_training import temporal_train_test_split, train_model, evaluate_model

def main():
    # Step 1: Define tickers and download filings
    tickers = [
        # Technology
        "AAPL",
        
        # Finance
        "JPM",
        
        # Healthcare
        "JNJ",
        
        # Consumer Discretionary
        "AMZN",
        
        # Energy
        "XOM",
        
        # Utilities
        "DUK",
        
        # Consumer Staples
        "PG",
        
        # Industrial
        "BA",
        
        # Telecommunications
        "VZ",
        
        # Real Estate
        "SPG",
    ]  # Add more tickers as needed
    download_filings(tickers, num_filings=2)
    
    # Step 2: Collect and align filings with stock data
    filings = collect_filings(tickers)
    filings = align_data(filings)
    filings = [f for f in filings if f['Volatility'] is not None and f['Text']]
    
    if not filings:
        print("No filings with relevant text found. Exiting.")
        return

    # Step 3: Preprocess text data
    filings = preprocess_filings(filings)
    
    # Step 4: Convert filings to DataFrame
    filings_df = pd.DataFrame(filings)
    
    # Step 5: Feature extraction
    X, vectorizer = extract_features(filings)
    
    # Step 6: Prepare target variable
    y = filings_df['Volatility'].values
    
    # Step 7: Train-test split
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
    
    # Step 8: Train the model
    model = train_model(X_train, y_train)
    
    # Step 9: Evaluate the model
    y_pred, mae, rmse = evaluate_model(model, X_test, y_test)
    print("y_test", y_test)
    print("y_pred", y_pred)
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    
    # Step 10: Plot the results
    plt.figure(figsize=(10,6))
    plt.plot(range(len(y_test)), y_test, label='Actual Volatility')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Volatility')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Volatility')
    plt.title('Actual vs. Predicted Volatility')
    plt.show()

if __name__ == "__main__":
    main()

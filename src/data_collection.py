import os
import glob
import time
import re
from datetime import datetime, timedelta
import logging

from sec_edgar_downloader import Downloader
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import parse_filing, clean_text

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def download_filings(tickers, num_filings=4, download_folder="sec-edgar-filings"):
    dl = Downloader(download_folder, "masonchoi@umass.edu")
    for ticker in tickers:
        time.sleep(1)  # Sleep for 1 second to avoid rate limiting
        dl.get("10-Q", ticker, limit=num_filings)
    logging.info(f"Downloaded filings for tickers: {tickers}")

def extract_filing_date(file_path):
    date_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')  # YYYY-MM-DD format
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = date_pattern.search(line)
                if match:
                    filing_date = match.group(0)
                    return datetime.strptime(filing_date, '%Y-%m-%d')
    except Exception as e:
        logging.error(f"Error extracting filing date from {file_path}: {e}")
    return None

def collect_filings(tickers, download_folder="sec-edgar-filings"):
    filings = []
    for ticker in tickers:
        filing_dirs = glob.glob(f"{download_folder}/{ticker}/10-Q/*")
        for filing_dir in filing_dirs:
            if os.path.isdir(filing_dir):
                file_path = os.path.join(filing_dir, "full-submission.txt")
                if not os.path.isfile(file_path):
                    logging.warning(f"No submission file found in directory: {filing_dir}")
                    continue

                filing_date = extract_filing_date(file_path)
                if not filing_date:
                    logging.warning(f"Could not find a valid date for filing: {file_path}. Skipping.")
                    continue

                text = parse_filing(file_path)
                if not text:
                    logging.warning(f"No relevant text extracted from filing: {file_path}. Skipping.")
                    continue

                text = clean_text(text)

                filings.append({
                    "Ticker": ticker,
                    "Filing Date": filing_date,
                    "Text": text
                })
    return filings

def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        return stock
    except Exception as e:
        logging.error(f"Error downloading stock data for {ticker} from {start_date} to {end_date}: {e}")
        return None

def calculate_volatility(stock_data):
    if stock_data is None or stock_data.empty:
        return None
    if 'Adj Close' not in stock_data.columns:
        logging.warning("Stock data does not contain 'Adj Close'. Cannot compute volatility.")
        return None

    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    if stock_data['Daily Return'].count() < 2:
        # Not enough data points to compute a meaningful volatility
        logging.warning("Not enough data points to compute volatility.")
        return None

    volatility = stock_data['Daily Return'].std() * (252 ** 0.5)  # Annualized volatility
    return volatility

def normalize_values(values):
    """
    Normalize a list of volatility values using StandardScaler.
    Returns the normalized values and the scaler.
    """
    scaler = StandardScaler()
    reshaped = np.array(values).reshape(-1, 1)
    scaled_values = scaler.fit_transform(reshaped).flatten()
    return scaled_values, scaler

def align_data(filings, normalize_volatility=False):
    all_volatilities = []
    for filing in filings:
        ticker = filing['Ticker']
        filing_date = filing['Filing Date']
        start_date = filing_date
        end_date = filing_date + timedelta(days=90)

        stock_data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if stock_data is None or stock_data.empty:
            logging.warning(f"No stock data available for {ticker} from {start_date} to {end_date}. Setting Volatility=None.")
            filing['Volatility'] = None
            all_volatilities.append(None)
            continue

        vol = calculate_volatility(stock_data)
        filing['Volatility'] = vol
        all_volatilities.append(vol)

    # If normalization is requested, normalize the volatility values (ignoring None)
    if normalize_volatility:
        vol_values = [v for v in all_volatilities if v is not None]
        if vol_values:
            normalized_values, scaler = normalize_values(vol_values)
            # Assign normalized values back to filings in the correct order
            norm_index = 0
            for f in filings:
                if f['Volatility'] is not None:
                    f['Volatility'] = normalized_values[norm_index]
                    norm_index += 1
            logging.info("Volatility values normalized using StandardScaler.")
        else:
            logging.warning("No valid volatility values found for normalization.")

    return filings

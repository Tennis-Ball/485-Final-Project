import os
import glob
import time
import re
from datetime import datetime, timedelta

from sec_edgar_downloader import Downloader
import yfinance as yf
from bs4 import BeautifulSoup

from utils import parse_filing, clean_text

def download_filings(tickers, num_filings=4, download_folder="sec-edgar-filings"):
    dl = Downloader(download_folder, "masonchoi@umass.edu")
    for ticker in tickers:
        time.sleep(1)  # Sleep for 1 second to avoid rate limiting
        dl.get("10-Q", ticker, limit=num_filings)
    print("Downloaded filings for tickers:", tickers)

def extract_filing_date(file_path):
    # Use a regular expression to search for date pattern in the content
    date_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')  # YYYY-MM-DD format
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = date_pattern.search(line)
            if match:
                filing_date = match.group(0)
                return datetime.strptime(filing_date, '%Y-%m-%d')
    return None

def collect_filings(tickers, download_folder="sec-edgar-filings"):
    filings = []
    for ticker in tickers:
        filing_dirs = glob.glob(f"{download_folder}/{ticker}/10-Q/*")
        for filing_dir in filing_dirs:
            # Check if filing_dir is a directory
            if os.path.isdir(filing_dir):
                # Look for a specific text file within the directory (e.g., full-submission.txt)
                file_path = os.path.join(filing_dir, "full-submission.txt")
                if not os.path.isfile(file_path):
                    print(f"No submission file found in directory: {filing_dir}")
                    continue
                
                # Extract filing date
                filing_date = extract_filing_date(file_path)
                if not filing_date:
                    print(f"Could not find a date for filing: {file_path}")
                    continue
                
                # Parse and clean the filing content
                text = parse_filing(file_path)
                if not text:
                    continue  # Skip if no relevant text found
                text = clean_text(text)
                
                # Add filing data to the list
                filings.append({
                    "Ticker": ticker,
                    "Filing Date": filing_date,
                    "Text": text
                })
    return filings

def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

def calculate_volatility(stock_data):
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    volatility = stock_data['Daily Return'].std() * (252 ** 0.5)  # Annualized volatility
    return volatility

def align_data(filings):
    for filing in filings:
        ticker = filing['Ticker']
        filing_date = filing['Filing Date']
        start_date = filing_date
        end_date = filing_date + timedelta(days=90)
        stock_data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if stock_data.empty:
            filing['Volatility'] = None
            continue
        volatility = calculate_volatility(stock_data)
        filing['Volatility'] = volatility
    return filings

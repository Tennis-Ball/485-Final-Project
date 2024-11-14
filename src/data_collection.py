import os
import glob
import time
from datetime import datetime, timedelta

from sec_edgar_downloader import Downloader
import yfinance as yf

from utils import parse_filing, clean_text

def download_filings(tickers, num_filings=4, download_folder="SEC-Edgar-Data"):
    dl = Downloader(download_folder, "masonchoi@umass.edu")
    for ticker in tickers:
        time.sleep(1)  # Sleep for 1 second to avoid rate limiting
        dl.get("10-Q", ticker, limit=num_filings)
    print("Downloaded filings for tickers:", tickers)

def collect_filings(tickers, download_folder="SEC-Edgar-Data"):
    filings = []
    for ticker in tickers:
        filing_paths = glob.glob(f"{download_folder}/{ticker}/10-Q/*")
        for path in filing_paths:
            filing_date_str = os.path.basename(path)[:10]
            filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')
            text = parse_filing(path)
            if not text:
                continue  # Skip if no relevant text found
            text = clean_text(text)
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

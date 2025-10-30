import yfinance as yf
import pandas as pd

def load_stock_data(ticker: str, start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance and return a cleaned DataFrame.
    Columns returned: Date, Open, High, Low, Close, Adj Close, Volume
    """
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return pd.DataFrame()

    # Reset index
    data = data.reset_index()

    # 🔥 Flatten multi-level columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    return data

# 📈 Stock Price Predictor

A Streamlit web app that:
- Fetches stock data from Yahoo Finance
- Engineers simple features
- Trains a RandomForest model
- Predicts the next-day close price
- Plots the historical price

## Run locally
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

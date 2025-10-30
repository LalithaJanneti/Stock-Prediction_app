import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_stock_data
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import plotly.express as px
import joblib

# ----------------------------
# App config & CSS
# ----------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")


# Inject CSS
st.markdown(
"""
<style>
/* App background and card styling */
.stApp {
background: linear-gradient(180deg, #071226 0%, #0F1720 100%);
color: #e6f6ee; 
}


/* Title style */
.title {
color: #F0F9F4;
font-size: 28px;
font-weight: 700;
text-align: left;
}


/* Sidebar tweaks */
.stSidebar .stButton>button {
border-radius: 8px;
padding: 8px 12px;
}


/* Dataframe style */
.element-container .stDataFrame table {
background-color: #0b1220;
}


/* Small cards */
.prediction {
background: rgba(255,255,255,0.03);
padding: 12px;
border-radius: 8px;
}
</style>
""",
unsafe_allow_html=True,
)

# ----------------------------
# Sidebar: Inputs
# ----------------------------
st.sidebar.header("Stock & Date Selection")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, TCS.NS)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date (optional)", value=pd.to_datetime(pd.Timestamp("today").date()))


if isinstance(end_date, pd.Timestamp):
  end_str = end_date.strftime("%Y-%m-%d")
else:
  end_str = None


st.title("📈 Stock Price Predictor")
st.markdown("---")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def get_data(ticker, start, end):
  return load_stock_data(ticker, start=start, end=end)

data = get_data(ticker, start_date.strftime("%Y-%m-%d"), end_str)

if data.empty:
  st.error("No data found for the given ticker or date range. Please check the ticker symbol.")
  st.stop()

# Show basic info
st.subheader(f"Data for {ticker}")
col1, col2 = st.columns([3,1])
with col1:
  st.write(data.tail())
with col2:
  st.metric("Rows", len(data))
  st.metric("From", str(data['Date'].iloc[0].date()))
  st.metric("To", str(data['Date'].iloc[-1].date()))

# ----------------------------
# Visualize historical close price
# ----------------------------
fig = px.line(data, x='Date', y='Close', title=f"{ticker} - Closing Price")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Feature engineering
# ----------------------------
df = data.copy()
df['Return'] = df['Close'].pct_change()
df['MA7'] = df['Close'].rolling(7).mean()
df['MA30'] = df['Close'].rolling(30).mean()
df['Prev_Close'] = df['Close'].shift(1)
# Label for classification: 1 if next-day close > today close, else 0
df['Target_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)


# Drop na
df = df.dropna().reset_index(drop=True)

st.subheader("Feature Preview")
st.write(df[['Date', 'Close', 'Prev_Close', 'MA7', 'MA30', 'Return', 'Target_Up']].tail())

# ----------------------------
# Prepare features and split
# ----------------------------
FEATURES = ['Prev_Close', 'MA7', 'MA30', 'Return']
X = df[FEATURES]
y_reg = df['Close']
y_clf = df['Target_Up']

# Sequential train/test split (80/20)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_reg_train, y_reg_test = y_reg.iloc[:split_index], y_reg.iloc[split_index:]
y_clf_train, y_clf_test = y_clf.iloc[:split_index], y_clf.iloc[split_index:]

# ----------------------------
# Train models
# ----------------------------
@st.cache_data
def train_models(X_tr, y_reg_tr, y_clf_tr):
   rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
   rf_reg.fit(X_tr, y_reg_tr)

   rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
   rf_clf.fit(X_tr, y_clf_tr)

   return rf_reg, rf_clf

model_reg, model_clf = train_models(X_train, y_reg_train, y_clf_train)

# ----------------------------
# Evaluate
# ----------------------------
reg_pred = model_reg.predict(X_test)
clf_pred = model_clf.predict(X_test)

mse = mean_squared_error(y_reg_test, reg_pred)
rmse = np.sqrt(mse)
acc = accuracy_score(y_clf_test, clf_pred)
cm = confusion_matrix(y_clf_test, clf_pred)

st.subheader("Model Performance on Test Set")
col_a, col_b = st.columns(2)
with col_a:
   st.metric("Regression RMSE", f"{rmse:.4f}")
   st.write(f"MSE: {mse:.4f}")
with col_b:
   st.metric("Classification Accuracy", f"{acc:.4f}")
   st.write("Confusion Matrix:")
   st.write(cm)

# Plot actual vs predicted for regression
res_df = pd.DataFrame({'Date': df['Date'].iloc[split_index:].values, 'Actual': y_reg_test.values, 'Predicted': reg_pred})
fig2 = px.line(res_df, x='Date', y=['Actual', 'Predicted'], title='Actual vs Predicted Close Price')
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Next-day prediction
# ----------------------------
st.subheader("🔮 Next Day Prediction")
last_row = X.iloc[-1].values.reshape(1, -1)
next_price_pred = model_reg.predict(last_row)[0]
next_dir_pred = model_clf.predict(last_row)[0]

col_n1, col_n2 = st.columns(2)
with col_n1:
   st.markdown("<div class='prediction'>", unsafe_allow_html=True)
   st.write(f"**Last Close:** {df['Close'].iloc[-1]:.2f}")
   st.write(f"**Predicted Next Close:** {next_price_pred:.2f}")
   st.write(f"**Predicted Direction:** {'UP' if next_dir_pred==1 else 'DOWN'}")
   st.markdown("</div>", unsafe_allow_html=True)

with col_n2:
   st.write("### Prediction Inputs")
   st.write({k: float(v) for k, v in zip(FEATURES, X.iloc[-1].tolist())})

# ----------------------------
# Save models option
# ----------------------------
if st.button('Save Models (to models/ folder)'):
   joblib.dump(model_reg, 'models/rf_reg.pkl')
   joblib.dump(model_clf, 'models/rf_clf.pkl')
   st.success('Models saved to models/rf_reg.pkl and models/rf_clf.pkl')

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with yfinance, scikit-learn, and Streamlit. For educational/demo use only — not financial advice.")













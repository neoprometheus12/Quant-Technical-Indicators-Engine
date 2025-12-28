import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Quant Technical Indicators Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# TITLE
# =========================
st.title("Quant Technical Indicators Engine")
st.caption("Built from scratch | Fully vectorized | Transparent indicator mechanics")

# =========================
# SIDEBAR — MARKET SELECTION
# =========================
st.sidebar.header("Market Selection")

ticker = st.sidebar.text_input("Ticker (Stock / Index / Crypto)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# =========================
# SIDEBAR — INDICATOR TOGGLES
# =========================
st.sidebar.header("Indicators")

use_sma = st.sidebar.checkbox("Simple Moving Average (SMA)", True)
use_ema = st.sidebar.checkbox("Exponential Moving Average (EMA)", True)
use_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", True)
use_macd = st.sidebar.checkbox("MACD", True)
use_bb = st.sidebar.checkbox("Bollinger Bands", True)

# =========================
# SIDEBAR — PARAMETERS
# =========================
st.sidebar.header("Parameters")

sma_window = st.sidebar.slider("SMA Window", 5, 200, 20)
ema_window = st.sidebar.slider("EMA Window", 5, 200, 20)

rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought Level", 60, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold Level", 10, 40, 30)

macd_fast = st.sidebar.slider("MACD Fast EMA", 5, 20, 12)
macd_slow = st.sidebar.slider("MACD Slow EMA", 20, 50, 26)
macd_signal = st.sidebar.slider("MACD Signal EMA", 5, 20, 9)

bb_window = st.sidebar.slider("Bollinger Window", 10, 50, 20)
bb_std = st.sidebar.slider("Bollinger Standard Deviation", 1.0, 3.0, 2.0)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data available. Check ticker or date range.")
    st.stop()

price = df["Close"]

# =========================
# INDICATOR FUNCTIONS
# =========================
def SMA(series, n):
    return series.rolling(n).mean()

def EMA(series, n):
    return series.ewm(span=n, adjust=False).mean()

def RSI(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def MACD(series, fast, slow, signal):
    macd = EMA(series, fast) - EMA(series, slow)
    signal_line = EMA(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def Bollinger(series, n, k):
    mid = SMA(series, n)
    std = series.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower

# =========================
# CALCULATIONS
# =========================
if use_sma:
    df["SMA"] = SMA(price, sma_window)

if use_ema:
    df["EMA"] = EMA(price, ema_window)

if use_rsi:
    df["RSI"] = RSI(price, rsi_window)

if use_macd:
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = MACD(
        price, macd_fast, macd_slow, macd_signal
    )

if use_bb:
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = Bollinger(
        price, bb_window, bb_std
    )

# =========================
# SIGNAL LOGIC
# =========================
df["Signal"] = "HOLD"

if use_rsi:
    df.loc[df["RSI"] < rsi_oversold, "Signal"] = "BUY"
    df.loc[df["RSI"] > rsi_overbought, "Signal"] = "SELL"

# =========================
# PRICE CHART
# =========================
st.subheader("Price and Trend Indicators")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(price, label="Close", color="black")

if use_sma:
    ax.plot(df["SMA"], label="SMA", linestyle="--")
if use_ema:
    ax.plot(df["EMA"], label="EMA", linestyle="--")
if use_bb:
    ax.plot(df["BB_Upper"], alpha=0.5)
    ax.plot(df["BB_Lower"], alpha=0.5)

ax.legend()
st.pyplot(fig)

# =========================
# RSI PANEL
# =========================
if use_rsi:
    st.subheader("Relative Strength Index (RSI)")
    st.markdown("""
RSI Interpretation:
- Values below the oversold threshold indicate potential mean reversion.
- Values above the overbought threshold indicate potential exhaustion.
""")

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(df["RSI"], color="purple")
    ax.axhline(rsi_overbought, linestyle="--", color="red")
    ax.axhline(rsi_oversold, linestyle="--", color="green")
    st.pyplot(fig)

# =========================
# MACD PANEL
# =========================
if use_macd:
    st.subheader("MACD Momentum Analysis")
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(df["MACD"], label="MACD")
    ax.plot(df["MACD_Signal"], label="Signal Line")
    ax.bar(df.index, df["MACD_Hist"], alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# =========================
# SIGNAL TABLE
# =========================
st.subheader("Latest Trading Signals")
st.dataframe(
    df[["Close", "Signal"]].tail(15)
)

# =========================
# DOWNLOAD
# =========================
csv = df.to_csv().encode()
st.download_button(
    "Download Full Indicator Data",
    csv,
    "technical_signals.csv",
    "text/csv"
)

# =========================
# FOOTER
# =========================
st.markdown("""
---
This dashboard demonstrates indicator engineering, parameter sensitivity, 
and transparent signal generation suitable for quantitative research workflows.
""")

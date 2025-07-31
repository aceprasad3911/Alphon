# alpha_vantage_api_calls.py

# Raw Data Calls from API
 
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key

###########################################
### 1. Core Time Series Stock Data APIs ###
###########################################
import requests

# Intraday Data (TIME_SERIES_INTRADAY) ---------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Daily Data (TIME_SERIES_DAILY) ---------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Daily Adjusted Data (TIME_SERIES_DAILY_ADJUSTED) ---------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Weekly Data (TIME_SERIES_WEEKLY) -------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Weekly Adjusted Data (TIME_SERIES_WEEKLY_ADJUSTED) -------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Monthly Data (TIME_SERIES_MONTHLY) -----------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Monthly Adjusted Data (TIME_SERIES_MONTHLY_ADJUSTED) -----------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Quote Endpoint (GLOBAL_QUOTE) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Realtime Bulk Quotes (REALTIME_BULK_QUOTES) --------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=REALTIME_BULK_QUOTES&symbol=MSFT,AAPL,IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Search Endpoint (SYMBOL_SEARCH) --------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=tesco&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Market Open & Close Status (MARKET_STATUS) --------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MARKET_STATUS&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

############################
### 2. Options Data APIs ###
############################
import requests

# Realtime Options (REALTIME_OPTIONS) ----------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Historical Options (HISTORICAL_OPTIONS) ------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

##############################
### 3. Alpha Intelligence™ ###
##############################
import requests

# Market News & Sentiment (NEWS_SENTIMENT) -----------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Earnings Call Transcript (EARNINGS_CALL_TRANSCRIPT)
url = 'https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=IBM&quarter=2024Q1&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Top Gainers, Losers, and Most Actively Traded Tickers (TOP_GAINERS_LOSERS) -------------------------------------------
url = 'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Insider Transactions (INSIDER_TRANSACTIONS) --------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Advanced Analytics (Fixed Window) (ANALYTICS_FIXED_WINDOW) -----------------------------------------------------------
url = 'https://alphavantageapi.co/timeseries/analytics?SYMBOLS=AAPL,MSFT,IBM&RANGE=2023-07-01&RANGE=2023-08-31&INTERVAL=DAILY&OHLC=close&CALCULATIONS=MEAN,STDDEV,CORRELATION&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Advanced Analytics (Sliding Window) (ANALYTICS_SLIDING_WINDOW) -------------------------------------------------------
url = 'https://alphavantageapi.co/timeseries/running_analytics?SYMBOLS=AAPL,IBM&RANGE=2month&INTERVAL=DAILY&OHLC=close&WINDOW_SIZE=20&CALCULATIONS=MEAN,STDDEV(annualized=True)&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

########################
### Fundamental Data ###
########################
import requests
import csv

# Company Overview (OVERVIEW) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# ETF Profile & Holdings (ETF_PROFILE) ---------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ETF_PROFILE&symbol=QQQ&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Corporate Action - Dividends (DIVIDENDS) -----------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=DIVIDENDS&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Corporate Action - Splits (SPLITS) -----------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=SPLITS&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Income Statement (INCOME_STATEMENT) ----------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Balance Sheet (BALANCE_SHEET) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Cash Flow (CASH_FLOW)-------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=CASH_FLOW&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Earnings History (EARNINGS) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Earnings Estimates (EARNINGS_ESTIMATES) ------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=EARNINGS_ESTIMATES&symbol=IBM&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Listing & Delisting Status (LISTING_STATUS)
CSV_URL = 'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo'
with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)

# Earnings Calendar (EARNINGS_CALENDAR) --------------------------------------------------------------------------------
CSV_URL = 'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=demo'
with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)

# IPO Calendar (IPO_CALENDAR) ------------------------------------------------------------------------------------------
CSV_URL = 'https://www.alphavantage.co/query?function=IPO_CALENDAR&apikey=demo'
with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)

######################################
### 5. Foreign Exchange Rates (FX) ###
######################################
import requests

# Currency Exchange Rate (CURRENCY_EXCHANGE_RATE) ----------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=JPY&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# FX Intraday (FX_INTRADAY) --------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=EUR&to_symbol=USD&interval=5min&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# FX Daily (FX_DAILY) --------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# FX Weekly (FX_WEEKLY) ------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol=EUR&to_symbol=USD&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

######################################
### 6. Digital & Crypto Currencies ###
######################################
import requests

# Crypto Intraday (CRYPTO_INTRADAY) ------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol=ETH&market=USD&interval=5min&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Digital Currency Daily (DIGITAL_CURRENCY_DAILY) ----------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Digital Currency Weekly (DIGITAL_CURRENCY_WEEKLY) --------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_WEEKLY&symbol=BTC&market=EUR&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Digital Currency Monthly (DIGITAL_CURRENCY_MONTHLY) ------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_MONTHLY&symbol=BTC&market=EUR&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

######################
### 7. Commodities ###
######################
import requests

# Crude Oil Prices: West Texas Intermediate (WTI) ----------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=WTI&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Crude Oil Prices (Brent) (BRENT) -------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=BRENT&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Natural Gas (NATURAL_GAS) --------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=NATURAL_GAS&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price of Copper (COPPER) --------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=COPPER&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price of Aluminum (ALUMINUM) ----------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ALUMINUM&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price of Wheat (WHEAT) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=WHEAT&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price of Corn (CORN) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=CORN&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price of Cotton (COTTON) ---------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=COTTON&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price of Sugar (SUGAR) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=SUGAR&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price of Coffee (COFFEE) --------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=COFFEE&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Global Price Index of All Commodities (ALL_COMMODITIES) --------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ALL_COMMODITIES&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

##############################
### 8. Economic Indicators ###
##############################
import requests

# Real GDP (REAL_GDP) --------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Real GDP Per Capita (REAL_GDP_PER_CAPITA) ----------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Treasury Yield (TREASURY_YIELD) --------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Federal Funds Rate (FEDERAL_FUNDS_RATE) ------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# CPI (CPI) ------------------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Inflation (INFLATION) ------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=INFLATION&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Retail Sales (RETAIL_SALES) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=RETAIL_SALES&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Durables (DURABLES) --------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=DURABLES&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Unemployment (UNEMPLOYMENT) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Nonfarm Payroll (NONFARM_PAYROLL) ------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

#################################
### # 9. Technical Indicators ###
#################################

# Simple Moving Average (SMA) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=SMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Exponential Moving Average (EMA) -------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=EMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Weighted Moving Average (WMA) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=WMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Double Exponential Moving Average (DEMA) -----------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=DEMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Triple Exponential Moving Average (TEMA) -----------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TEMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Triangular Moving Average (TRIMA) ------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TRIMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Kaufman Adaptive Moving Average (KAMA) -------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=KAMA&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# MESA Adaptive Moving Average (MAMA) ----------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MAMA&symbol=IBM&interval=daily&series_type=close&fastlimit=0.02&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Volume Weighted Average Price (VWAP) ---------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=VWAP&symbol=IBM&interval=15min&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Triple Exponential Moving Average (T3) -------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=T3&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Moving Average Convergence / Divergence (MACD) -----------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MACD&symbol=IBM&interval=daily&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Moving Average Convergence / Divergence with Controllable Moving Average Type (MACDEXT) ------------------------------
url = 'https://www.alphavantage.co/query?function=MACDEXT&symbol=IBM&interval=daily&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Stochastic Oscillator (STOCH) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=STOCH&symbol=IBM&interval=daily&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Stochastic Fast (STOCHF) ---------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=STOCHF&symbol=IBM&interval=daily&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Relative Strength Index (RSI) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=RSI&symbol=IBM&interval=weekly&time_period=10&series_type=open&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Stochastic Relative Strength Index (STOCHRSI) ------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=STOCHRSI&symbol=IBM&interval=daily&time_period=10&series_type=close&fastkperiod=6&fastdmatype=1&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Williams' %R (WILLR) -------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=WILLR&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Average Directional Movement Index (ADX) -----------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ADX&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Average Directional Movement Index Rating (ADXR) ---------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ADXR&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Absolute Price Oscillator (APO) --------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=APO&symbol=IBM&interval=daily&series_type=close&fastperiod=10&matype=1&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Percentage Price Oscillator (PPO) ------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=PPO&symbol=IBM&interval=daily&series_type=close&fastperiod=10&matype=1&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Momentum (MOM) -------------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MOM&symbol=IBM&interval=daily&time_period=10&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Balance of Power (BOP) -----------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=BOP&symbol=IBM&interval=daily&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Commodity Channel Index (CCI) ----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=CCI&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Chande Momentum Oscillator (CMO) -------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=CMO&symbol=IBM&interval=weekly&time_period=10&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Rate of Change (ROC) -------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ROC&symbol=IBM&interval=weekly&time_period=10&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Rate of Change Ratio (ROCR) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ROCR&symbol=IBM&interval=daily&time_period=10&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Aroon (AROON) --------------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=AROON&symbol=IBM&interval=daily&time_period=14&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Aroon Oscillator (AROONOSC) ------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=AROONOSC&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Money Flow Index (MFI) -----------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MFI&symbol=IBM&interval=weekly&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# TRIX -----------------------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TRIX&symbol=IBM&interval=daily&time_period=10&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Ultimate Oscillator (ULTOSC) -----------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ULTOSC&symbol=IBM&interval=daily&timeperiod1=8&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Directional Movement Index (DX) --------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=DX&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Minus Directional Indicator (MINUS_DI) -------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MINUS_DI&symbol=IBM&interval=weekly&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Plus Directional Indicator (PLUS_DI) ---------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=PLUS_DI&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Minus Directional Movement (MINUS_DM) --------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MINUS_DM&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Plus Directional Movement (PLUS_DM) -----------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=PLUS_DM&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Bollinger Bands (BBANDS) ---------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=BBANDS&symbol=IBM&interval=weekly&time_period=5&series_type=close&nbdevup=3&nbdevdn=3&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Midpoint (MIDPOINT) --------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MIDPOINT&symbol=IBM&interval=daily&time_period=10&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Midprice (MIDPRICE) --------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=MIDPRICE&symbol=IBM&interval=daily&time_period=10&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Parabolic SAR (SAR) --------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=SAR&symbol=IBM&interval=weekly&acceleration=0.05&maximum=0.25&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# True Range (TRANGE) --------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=TRANGE&symbol=IBM&interval=daily&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Average True Range (ATR) ---------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ATR&symbol=IBM&interval=daily&time_period=14&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Normalized Average True Range (NATR) ---------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=NATR&symbol=IBM&interval=weekly&time_period=14&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Chaikin A/D Line (AD) ------------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=AD&symbol=IBM&interval=daily&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Chaikin A/D Oscillator (ADOSC) ---------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=ADOSC&symbol=IBM&interval=daily&fastperiod=5&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# On Balance Volume (OBV) ----------------------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=OBV&symbol=IBM&interval=weekly&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Hilbert Transform, Instantaneous Trendline (HT_TRENDLINE) ------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=HT_TRENDLINE&symbol=IBM&interval=daily&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Hilbert Transform, Sine Wave (HT_SINE) -------------------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=HT_SINE&symbol=IBM&interval=daily&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Hilbert Transform, Trend vs Cycle Mode (HT_TRENDMODE) ----------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=HT_TRENDMODE&symbol=IBM&interval=weekly&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Hilbert Transform, Dominant Cycle Period (HT_DCPERIOD) ---------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=HT_DCPERIOD&symbol=IBM&interval=daily&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Hilbert Transform, Dominant Cycle Phase (HT_DCPHASE) -----------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=HT_DCPHASE&symbol=IBM&interval=daily&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)

# Hilbert Transform, Phasor Components (HT_PHASOR) ---------------------------------------------------------------------
url = 'https://www.alphavantage.co/query?function=HT_PHASOR&symbol=IBM&interval=weekly&series_type=close&apikey=demo'
r = requests.get(url)
data = r.json()
print(data)
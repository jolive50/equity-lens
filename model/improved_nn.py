import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
import keras
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pipelines.batch.historical import YahooFinanceConfig, fetch_yahoo_prices

class StockForecastRNN():
    scaler = StandardScaler()
    model = Sequential()

    def __init__(self, data):
        self.scaler.fit_transform(data)
        self.model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(data.shape[1], 1)))
        self.model.add(SimpleRNN(units=50, return_sequences=False))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

today = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() + timedelta(-7)).strftime('%Y-%m-%d')
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
yf_config = YahooFinanceConfig(tickers, start_date, today, "1m")
fetch_yahoo_prices(yf_config)
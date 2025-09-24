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

def predict_stock(ticker):
    # Get data from yfinance
    today = datetime.today().strftime('%Y-%m-%d')
    raw_data = yf.download(ticker, start='2020-01-01', end=today)

    # Pre-process data for use with RNN, setting time step
    TIME_STEP=60

    # Keep only closing price for training (for now)
    data = raw_data['Close'].values.reshape(-1, 1)

    # Scale data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data) - TIME_STEP - 1):
        X.append(data[i:i+TIME_STEP, 0])
        y.append(data[i+TIME_STEP, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into train and test sets using 80/20 split
    train_size=int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build RNN
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train RNN
    model.fit(X_train, y_train, epochs=20, batch_size=64)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_unscaled, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, predictions)

    """
    print(f"Unscaled Mean Squared Error (MSE): {mse}")
    print(f"Unscaled Root Mean Squared Error (RMSE): {rmse}")
    print(f"Unscaled Mean Absolute Error (MAE): {mae}")

    plt.figure(figsize=(10,6))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Real Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    """

    start_date = (datetime.today() + timedelta(-60)).strftime('%Y-%m-%d')
    raw_data = yf.download(ticker, start=start_date, end=today)
    new_x = raw_data['Close'].values.reshape(-1, 1)
    new_x = scaler.transform(new_x)
    new_x = new_x.reshape(new_x.shape[0], new_x.shape[1], 1)
    y_pred = model.predict([new_x[-1]])
    y_pred = scaler.inverse_transform(y_pred)
    return float(y_pred[0])
    

company = 'AAPL'
pred = predict_stock(company)
print(f"Model predicts that price of {company} stock will be ${pred:.2f} tomorrow.")
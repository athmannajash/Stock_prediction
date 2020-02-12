import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import plotly.plotly as py
import plotly.graph_objs as go
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates

#function to plot candlestick chart stock price
def plot():
    plt.style.use('ggplot')
    data = pd.read_csv("NVDA.csv")
    data = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].apply(mpl_dates.date2num)
    data = data.astype(float)

    # Creating Subplots
    fig, ax = plt.subplots()

    candlestick_ohlc(ax, data.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    # Setting labels & titles
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle('NVIDIA STOCK PRICE CANDLESTICK CHART')

    # Formatting Date
    date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

#plot price and volume before normalizing
def plot_before():
    df = pd.read_csv("NVDA.csv")
    df = df[['Date','Open', 'High', 'Low', 'Close','Volume']]
    df.plot(x='Date', rot=0)
    plt.show()

#function to normalize data from csv
def process_data():
    data = pd.read_csv('NVDA.csv', date_parser = True)
    #assigning training data and test data 4:1
    data_train = data[data['Date']<='2019-01-01'].copy()
    data_test = data[data['Date']>'2019-01-01'].copy()

    #dropping columns date and ADJ close on both training and test data
    training_data = data_train.drop(['Date','Adj Close'], axis = 1)
    test_data = data_test.drop(['Date','Adj Close'], axis = 1)


    #Scaling data both training and test data
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    print("Scaled training data")
    print(training_data)

    test_data1 = scaler.fit_transform(test_data)
    print("Scaled test data")
    print(test_data1)

    #plotting scaled data
    scaled_data=pd.DataFrame({'Open': training_data[:, 0], 'High': training_data[:, 1], 'Low': training_data[:, 2], 'Close': training_data[:, 3],'Volume': training_data[:, 4]})
    data_train = data_train[['Date']]
    scaled_data = pd.concat([data_train, scaled_data], axis=1)
    print(scaled_data)
    scaled_data.plot(x='Date', rot=0)
    plt.show()


process_data()
plot()
plot_before()

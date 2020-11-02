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

    # Formatting Datemain4
    date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

plot()

'''=================================================PART-1======================================================'''
from nsepy import get_history
from datetime import date
import pandas as pd
# LOADING DATA

from nsepy.derivatives import get_expiry_date
expiry = get_expiry_date(year=2015, month=12)
print(expiry)

'''
tcs = get_history(symbol='TCS',
                   start=date(2015,1,1),
                   end=date(2015,12,31))
print(tcs)

#setting date as the starting index
tcs.insert(0, 'Date',  pd.to_datetime(tcs.index,format='%Y/%m/%d') )
a = type(tcs.index)
print(a)
b = type(tcs.Date)
print(b)
tcs.Date.dt
tcs.to_csv('tcs_stock.csv', encoding='utf-8', index=False)

"""================================For Infy================================"""

infy = get_history(symbol='INFY',
                   start=date(2015,1,1),
                   end=date(2015,12,31))
print(infy)

infy.insert(0, 'Date',  pd.to_datetime(infy.index,format='%Y/%m/%d'))

c = type(infy.index)
print(c)
d = type(infy.Date)
print(d)
infy.Date.dt

"""===============================For NIFTY==============================="""
nifty_it = get_history(symbol="NIFTYIT",
                            start=date(2015,1,1),
                            end=date(2015,12,31),
                            index=True)
print(nifty_it)
#setting date as starting index 
nifty_it.insert(0, 'Date',  pd.to_datetime(nifty_it.index,format='%Y-%m-%d') )

e = type(nifty_it.index)
print(e)
f = type(nifty_it.Date)
print(f)
nifty_it.Date.dt
'''
#import packages

from pandas import datetime
import numpy as np

TCS = pd.read_csv('tcs_stock.csv', parse_dates=['Date'])

INFY = pd.read_csv('infy_stock.csv', parse_dates=['Date'])

NIFTY_IT = pd.read_csv('nifty_it_index.csv', parse_dates=['Date'])


stocks = [TCS, INFY, NIFTY_IT]


TCS.name = 'TCS'
INFY.name = 'INFY'
NIFTY_IT.name = 'NIFTY_IT'

TCS["Date"] = pd.to_datetime(TCS["Date"])
INFY["Date"] = pd.to_datetime(INFY["Date"])
NIFTY_IT["Date"] = pd.to_datetime(NIFTY_IT["Date"])

a1 = TCS.head(10)
print(a1)
TCS.shape
c1 = INFY.head(10)
print(c1)
INFY.shape
e1 = NIFTY_IT.head(10)
print(e1)
NIFTY_IT.shape

# data extraction


def feature_build(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['Week_Of_Year'] = df.Date.dt.weekofyear
    
    
    
for i in range(len(stocks)):
    # print(stocks[i])
    feature_build(stocks[i])
    
TCS.shape
z = TCS.head(5)

# Defining a function for moving average with rolling window

def moving_average(series, n):
    """
        Calculating the average of last n observations
        n -> rolling window
    
    """
    return np.average(series[-n:])
weeks = [4, 16, 28, 40, 52]
def indexing(stock):
    stock.index = stock['Date']
    return stock
indexing(TCS)
indexing(INFY)
indexing(NIFTY_IT)

import matplotlib.pyplot as plt

def time_series(stock, weeks = [4, 16, 28, 40, 52]):
    
    dummy = pd.DataFrame()
    # First we do resampling into Weeks format to calculate for weeks
    dummy['Close'] = stock['Close'].resample('W').mean() 
     
    for i in range(len(weeks)):
        mean_avg = dummy['Close'].rolling(weeks[i]).mean() # M.A using inbuilt function
        dummy[" Mov.AVG for " + str(weeks[i])+ " Weeks"] = mean_avg
        print('Calculated Moving Averages: for {0} weeks: \n\n {1}' .format(weeks[i], dummy['Close']))
    ax = dummy.plot(title="Moving Averages for {} \n\n" .format(stock.name))
    ax.legend( loc ='upper right', prop={'size': 6})

time_series(TCS)
time_series(INFY)
time_series(NIFTY_IT)

TCS = TCS.asfreq('D', method ='pad')        # pad/ffill : forward-fill
INFY = INFY.asfreq('D', method ='pad')
NIFTY_IT = NIFTY_IT.asfreq('D', method ='pad')


TCS.name = 'TCS'
INFY.name = 'INFY'
NIFTY_IT.name = 'NIFTY_IT'

def roll_win(stock, win = [10, 75]):
    
    dummy = pd.DataFrame()
    
    dummy['Close'] = stock['Close']
     
    for i in range(len(win)):
        m_a = dummy['Close'].rolling(win[i]).mean() # M.A using predefined function
        dummy[" Mov.AVG for " + str(win[i])+ " Roll Window"] = m_a
        print('Calculated Moving Averages: for {0} weeks: \n\n {1}' .format(win[i], dummy['Close']))
    ax = dummy.plot(title="Moving Averages for {} \n\n" .format(stock.name))
    ax.legend( loc ='upper right',prop={'size': 6})
    
roll_win(TCS)
roll_win(INFY)
roll_win(NIFTY_IT)

# VOLUME SHOCKS

def vol_shocks(stock):
    """
    'Volume' - Vol_t
    'Volume next day - vol_t+1
    
    """
    stock["vol_t+1"] = stock.Volume.shift(1)  #next rows value
    
    stock["volume_shock"] = ((abs(stock["vol_t+1"] - stock["Volume"])/stock["Volume"]*100)  > 10).astype(int)
    
    return stock

vol_shocks(TCS)
vol_shocks(INFY)
vol_shocks(NIFTY_IT)

# PRICE SHOCKS

def price_shocks(stock):
    """
    'ClosePrice' - Close_t
    'Close Price next day - vol_t+1
    
    """
    stock["price_t+1"] = stock.Close.shift(1)  #next rows value
    
    stock["price_shock"] = (abs((stock["price_t+1"] - stock["Close"])/stock["Close"]*100)  > 2).astype(int)
    
    return stock


price_shocks(TCS)
price_shocks(INFY)
price_shocks(NIFTY_IT)

# PRICE BLACK SWAN

def black_swan(stock):
    
    stock["price_t+1"] = stock.Close.shift(1)
    
    stock["price_shock"] = (abs((stock["price_t+1"] - stock["Close"])/stock["Close"]*100)  > 2).astype(int)
    
    stock["price_black_swan"] = stock['price_shock'] # because it has same data anad info as price_shocks
    return stock

black_swan(TCS)
black_swan(INFY)
black_swan(NIFTY_IT)

def shock_wo_vol_shock(stock):
    
    stock["not_vol_shock"]  = (~(stock["volume_shock"].astype(bool))).astype(int)
    stock["price_shock_w/0_vol_shock"] = stock["not_vol_shock"] & stock["price_shock"]
    
    return stock

shock_wo_vol_shock(TCS)
shock_wo_vol_shock(INFY)
shock_wo_vol_shock(NIFTY_IT)

'''=================================================PART-2======================================================'''

# Importing import libraries
import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.io import show
from bokeh.palettes import Blues9,RdBu3
from bokeh.models import ColumnDataSource, CategoricalColorMapper, ContinuousColorMapper
from bokeh.palettes import Spectral11

def bokeh_plot(stock):
    data = dict(stock=stock['Close'], Date=stock.index)
    
    p = figure(plot_width=800, plot_height=250,  title = 'time series for {}' .format(stock.name), x_axis_type="datetime")
    p.line(stock.index, stock['Close'], color='blue', alpha=0.6)
    
    #show price shock w/o vol shock
    
    p.square(stock.index, stock.Close*stock["price_shock_w/0_vol_shock"], size=5, legend='price shock without volume shock')
    show(p)
output_file("Timeseries.html")

bokeh_plot(TCS)
bokeh_plot(INFY)
bokeh_plot(NIFTY_IT)

from statsmodels.tsa.stattools import acf, pacf

def draw_pacf_(stock):
    
    lags = 50

    x = list(range(lags))

    fig = figure(plot_height=700, title="Partial Autocorrelation Plot" .format(stock.name))

    partial_autocorr = pacf(stock["Close"], nlags=lags)
    fig.vbar(x=x, top=partial_autocorr, width=0.8)
    show(fig)
output_file("pacf.html")

draw_pacf_(TCS)
draw_pacf_(INFY)
draw_pacf_(NIFTY_IT)


import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split

data=yf.download("JEPQ", start="2023-07-12", end="2026-07-11")
data.columns = data.columns.droplevel('Ticker')
data.columns.name = None

#20 day window as the stace space.

data['SMA_20'] = data['Close'].rolling(window=20).mean()

delta=data['Close'].diff()
gain=(delta.where(delta>0,0)).rolling(window=14).mean()
loss=(-delta.where(delta<0,0)).rolling(window=14).mean()
rs=gain/loss
data['RSI_14']=100-(100/(1+rs))

data.dropna(inplace=True)

train,test=train_test_split(data, shuffle=False, test_size=0.2)

# print(train.head())
# print(test.head())

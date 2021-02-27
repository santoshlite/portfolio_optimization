#python librairies
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
plt.style.use('fivethirtyeight')

#get the symbols of stocks
assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

#assign weights to the stock
weights = np.array([0.2,0.2,0.2,0.2,0.2])

#get the stock/portofolio starting date
stockStartDate= '2013-01-01'

#get the stock ending date (today)
today = datetime.today().strftime('%Y-%m-%d')


#create a dataframe to store the adjusted close proce of the stocks
df = pd.DataFrame()
#store the adjusted close price of the stock into the df
for stock in assets:
  df[stock] = web.DataReader(stock, data_source='yahoo', start = stockStartDate, end=today)['Adj Close']

#Portfolio Optimization
#Calculate the expected return and the annualised sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#optimize for max sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = 15000)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))

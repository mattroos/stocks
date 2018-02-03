# package_data_nasdaq.py
#
# Read in .csv files from eoddata.com and create a single pytorch-friendly
# structure.
#
#
# CSV files have a header and are formatted like this:
#       Symbol,Date,Open,High,Low,Close,Volume
#       AAAP,12-Jan-2017,29.19,29.5,28.28,29.01,150600
#
#
# WARNING: The raw data files include dates in which the market was closed,
# and all fields contain the closing value of the day before.
# TODO: Remove the data for the closed days?
#
#
# Side thoughts on models:
# 1. Don't identify individual stocks. Just have N stocks as input over M days. This
#    may help with model generalization.  And the model training can use augmented data
#    by shufflings the order of the stocks.
# 1b. How to generalize the above so the total number of stocks in/out of the model
#     could be variable (e.g., if a stock got added or dropped from the exchange.) Just
#     use zeros as input and output for "missing" stocks?
# 2. Include a label that indicates day of the week.
# 3. Cross-validate by training on, say, two years of data, and then test the model
#    on the subsequent, say, one month of data.  Could also try retraining the model
#    every day, for use on the following data, but that could be overkill.
# 4. Might use stocks gains as input in addition to actual (log or linear) prices.
# 5. Should probably always use log-scale prices. Or use gain, normalized by first day's price.
# 6. Model takes in how many days of data for one day of output prediction?  Two weeks?
# 7. Output should be gain, not actual price.
# 8. Use skip connections so inputs are fed into every layer.
# 9. Use only larger stocks, based on average price*volume (though some volume data is 0/NaN)


import numpy as np 
import hickle
import sys
import time
import glob
import datetime

#sys.exit()  # don't accidentally write over data


dirDataRoot = '/Users/roosmj1/Data/Nasdaq/'
dirDataList = ['NASDAQ_2007', 'NASDAQ_2008', 'NASDAQ_2009', 'NASDAQ_2010', 'NASDAQ_2011', 'NASDAQ_2012', 'NASDAQ_2013', 'NASDAQ_2014', 'NASDAQ_2015', 'NASDAQ_2016', 'NASDAQ_2017']
dirDataList.sort()


# Get list of all symbols and dates
print('\nGetting list of all symbols...')
symbols = np.asarray([], dtype='str')
dates = np.asarray([], dtype='str')
for dir in dirDataList:
    print('\t%s' % (dir))
    file_list = glob.glob( dirDataRoot + dir + '/*.csv')
    file_list.sort()

    for datafile in file_list:
        d = np.genfromtxt(datafile, dtype=None, delimiter=',', names=True)
        symbols = np.unique(np.concatenate((symbols, d['Symbol'])))
        dates = np.append(dates, d['Date'][0])
print('Done.\n')


# Read raw numbers into array
print('Reading in price and volume data...')
nSym = len(symbols)
nDates = len(dates)
dayofweek = [datetime.datetime.strptime(d,'%d-%b-%Y').weekday() for d in dates] # 0==Mon, 1==Tues, ...
#datenum = datetime.datetime.utcfromtimestamp(datetime.datetime.strptime(d,'%d-%b-%Y'))
prices = np.nan * np.zeros((nSym,nDates,4), dtype=np.float32)
volume = np.nan * np.zeros((nSym,nDates), dtype=np.float32)

for dir in dirDataList:
    file_list = glob.glob( dirDataRoot + dir + '/*.csv')

    for datafile in file_list:
        d = np.genfromtxt(datafile, dtype=None, delimiter=',', names=True)
        date = d['Date'][0]
        ix_date = np.where(dates==date)[0]
        b_symbols = np.in1d(symbols, d['Symbol'], assume_unique=True)

        prices[b_symbols,ix_date,0] = d['Open']
        prices[b_symbols,ix_date,1] = d['High']
        prices[b_symbols,ix_date,2] = d['Low']
        prices[b_symbols,ix_date,3] = d['Close']
        volume[b_symbols,ix_date] = d['Volume']
print('Done.\n')


# Save results
data = {}
data['symbols'] = symbols
data['dates'] = dates
data['prices'] = prices
data['volume'] = volume
data['dayofweek'] = dayofweek
f = open('data.hkl', 'w')
hickle.dump(data, f)


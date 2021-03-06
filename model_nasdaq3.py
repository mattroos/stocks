# model_nasdaq1.py
#
# Build, train, and test model.
#
#
# Side thoughts on models:
# 1. Don't identify individual stocks. Just have N stocks as input over M days. This
#    may help with model generalization.  And the model training can use augmented data
#    by shufflings the order of the stocks.
# 1b. How to generalize the above so the total number of stocks in/out of the model
#     could be variable (e.g., if a stock got added or dropped from the exchange.) Just
#     use zeros as input and output for "missing" stocks?
# 1c. Need to have a model that shares weight. Shuffling is stupid.
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


# TODO: Outputs are all identical ofter only one i_iteration. Something wrong.
# Getting weights with nans after 1 i_iteration. Bad gradients, somehow.

# TODO: How to use larger batch size?

# TODO: For more regularization, drop some fraction of the stocks in each batch/iteration.


import numpy as np 
import hickle
import sys
import time
from torch.autograd import Variable as V
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats.mstats import spearmanr

from matplotlib.pyplot import *     # for easier command line usage

plt.ion()

## Set model parameters
layer_sizes = [10, 10, 1]
dropout = 0.1
batch_size = 1
n_iters = 20000
n_days_input = 5
n_i_iter_per_log = 100
learn_rate = 0.0001
n_days_test = 100  # number of days at end of data set to reserve for testing

################################################################
# Initialize some things, and define functions and classes
################################################################

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

b_use_cuda = torch.cuda.is_available()

def TT(x):
    # convert array/tensor to torch tensor and put on GPU if available
    if type(x)==np.ndarray:
        x = torch.from_numpy(x).contiguous()
    if b_use_cuda:
        x = x.cuda()
    return x


## Model class
class fc_layers(nn.Module):
    def __init__(self, n_symbols, n_features, output_sizes, dropout=0):
        super(fc_layers, self).__init__()
        # TODO: Allow for variable number of layers
        # TODO: Use batch-norm?

        self.n_symbols = n_symbols
        self.n_kernels = output_sizes[0]

        self.fc1 = nn.Linear(n_features, output_sizes[0], bias=True)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_sizes[0], output_sizes[1], bias=True)
        self.do2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(output_sizes[1], output_sizes[2], bias=True)

        self.LinearSumm1 = nn.Linear(output_sizes[0], output_sizes[0], bias=True)
        self.LinearSumm2 = nn.Linear(output_sizes[0], output_sizes[1], bias=True)

    def forward(self, input):
        output = self.fc1(input)
        output = F.relu(output)
        output = self.do1(output)

        output = self.fc2(output)

        '''
        # Not clear if this helps or not
        output_summ = self.LinearSumm1(output)
        output_summ = torch.mean(output_summ, 1, keepdim=True)   # mean over all stocks
        output_summ = F.relu(output_summ)
        output_summ = self.LinearSumm2(output_summ)
        output = output + output_summ
        '''

        output = F.relu(output)
        #output = self.do2(output)

        output = self.fc3(output)
        return output


# Load data
data_dict = hickle.load('data.hkl')
symbols = data_dict['symbols']
dates = data_dict['dates']
dayofweek = data_dict['dayofweek']
volume = data_dict['volume']
prices = data_dict['prices']
del data_dict

# Remove days in which the market was closed (determined as days
# for which all stocks had volume=0 or volume=nan)
ixClosed = np.where(np.all((volume==0) | np.isnan(volume), axis=0))[0]
prices = np.delete(prices, ixClosed, axis=1)
volume = np.delete(volume, ixClosed, axis=1)
dates = np.delete(dates, ixClosed)
dayofweek = np.delete(dayofweek, ixClosed)

# # Remove stocks for which median volume is low?
# vol_median = np.nanmedian(volume, axis=1)
# threshold = np.median(vol_median)
# ixKeep = np.where(vol_median > threshold)[0]
# prices = prices[ixKeep,:,:]
# volume = volume[ixKeep,:]
# symbols = symbols[ixKeep]

# Remove stocks with very low or very high stock price, which may indicate
# the data is bad (at any time?  average/median over time)?
mx = np.nanmax(np.nanmax(prices[:,:,:], axis=2), axis=1)
mn = np.nanmin(np.nanmin(prices[:,:,:], axis=2), axis=1)
bKeep =  (mn > 1.0) & (mx < 1000.0)
prices = prices[bKeep,:,:]
volume = volume[bKeep,:]
symbols = symbols[bKeep]

## Set some bad data to nan
# Bad stocks on 6/17/13: ANTH, CERN, CRVL, CSWL, CYTK, INBK, PRAA
badstocks = ['ANTH', 'CERN', 'CRVL', 'CSWL', 'CYTK', 'INBK', 'PRAA']
ixDate = np.where(dates=='17-Jun-2013')[0]
for stock in badstocks:
    ixStock = np.where(symbols==stock)[0]
    prices[ixStock,ixDate,:] = np.nan

# Some stocks didn't trade on an open market day (volume=0). Replace
# prices with nan, for those days, so they are not used for model training
# or testing.
ixZeroVol = np.where(volume==0)
prices[ixZeroVol[0], ixZeroVol[1], :] = np.nan

# Use price gains relative to previous close, not absolute dollars
prices[:, 1:, :] = prices[:, 1:, :] / prices[:, 0:-1, 3:4]
prices[:, 0:1, :] = prices[:, 0:1, :] / prices[:, 0:1, 0:1]   # Relative to open price, for first day of data

# Replace all zeros and +/-infs by nans
prices[prices==0] = np.nan
volume[volume==0] = np.nan
prices[np.isinf(prices)] = np.nan
volume[np.isinf(volume)] = np.nan

## Use log-scale prices. How to deal with zeros?
if True:
    # prices = np.log10(prices)
    prices = np.log(prices)
    prices[np.isinf(prices)] = np.nan
    null_val = 0
else:
    null_val = 1

# Normalize prices and/or volume?
# TODO: This is not valid for cross validation.
#prices = (prices - np.mean(prices)) / np.std(prices)

# Center dayofweek about zero
dayofweek = np.asarray(dayofweek, dtype=np.float32) - 2.0


# Build model
n_symbols = len(symbols)
n_dates = len(dates)
#layer_sizes = layer_sizes + [n_symbols]  # adding element to list, not adding value to elements
n_features = n_days_input * 4   # 4 is because we have prices for open, high, low, close
net = fc_layers(n_symbols, n_features, layer_sizes, dropout=dropout)
if b_use_cuda:
    net = net.cuda()


# Train model
optimizer = optim.Adam(net.parameters(), lr=learn_rate)

loss_history = np.zeros(n_iters)
loss_baseline = np.zeros(n_iters)
spear = np.zeros(n_iters)
# loss_train_history = np.asarray([])
loss_test_history = np.asarray([])

criterion = nn.MSELoss()

t_start = time.time()
for i_iter in range(n_iters):
    net.train()

    # Initialize for batch/i_iteration
    optimizer.zero_grad()

    # Build the batch
    ix_date_start = np.random.randint(0, n_dates - n_days_input - n_days_test, batch_size)
    batch_in = np.zeros((batch_size, n_symbols, n_days_input, 4))
    batch_out = np.zeros((batch_size, n_symbols))
    for iBatch in range(batch_size):
        batch_in[iBatch] = prices[:, ix_date_start[iBatch]:ix_date_start[iBatch]+n_days_input, :]
        batch_out[iBatch] = prices[:, ix_date_start[iBatch]+n_days_input, -1]   # closing price of next day after range of input days
    batch_in = np.reshape(batch_in, (batch_size, n_symbols, n_days_input*4))
    batch_out = np.reshape(batch_out, (batch_size, n_symbols))

    # if i_iter==3751:    # very low loss on this sample. why?
    #     sys.exit()

    # If any price for a symbol is nan, remove that symbol
    bKeep = np.logical_not(np.any(np.isnan(batch_in), axis=2))[0]
    batch_in = batch_in[:,bKeep,:]
    batch_out = batch_out[:,bKeep]

    bKeep = np.logical_not(np.any(np.isnan(batch_out), axis=0))
    batch_in = batch_in[:,bKeep,:]
    batch_out = batch_out[:,bKeep]

    # Convert to Variable and move to GPU if GPU available
    batch_in = V(TT(batch_in.astype(np.float32)))
    batch_out = V(TT(batch_out.astype(np.float32)))

    # Put data through model and compute loss
    output = net(batch_in)
    
    #loss = criterion(output, batch_out)
    
    # Use negative profit as the loss. Not sure the right way to normalize this to "unity neutral" investment.
    output = output / torch.norm(output, p=2, dim=1)
    output = output - torch.mean(output)
    loss = -torch.sum(torch.exp(batch_out) * output[:,:,0] - output[:,:,0])    # profit = SUM[gi*xi - xi], gi=linear gain, xi=investment (negative if short)
    # TODO: Add heavy L1 cost to promote small number of trades?


    # Compute gradients and update parameters
    loss.backward()
    optimizer.step()

    loss_history[i_iter] = loss.data.cpu().numpy()
    loss_baseline[i_iter] = criterion(TT(np.zeros(batch_out.size(),dtype=np.float32)), batch_out).data.cpu().numpy()
    spear[i_iter] = spearmanr(output[0,:,0].data.cpu().numpy(), batch_out[0,:].data.cpu().numpy())[0]


    if (i_iter+1)%n_i_iter_per_log==0 | (i_iter==0):
        print('Iteration %d, Loss=%0.5f, Duration=%0.3f' % (i_iter+1, loss.data.cpu().numpy(), time.time()-t_start))

        ## Put testing data through model
        net.eval()

        loss_day = np.asarray([])
        for i_test_samp in range(n_days_test-n_days_input):

            # Build the batch
            ix_date_start = np.asarray([n_dates - n_days_test + i_test_samp])
            batch_in = np.zeros((1, n_symbols, n_days_input, 4))
            batch_out = np.zeros((1, n_symbols))
            for iBatch in range(1):
                batch_in[iBatch] = prices[:, ix_date_start[iBatch]:ix_date_start[iBatch]+n_days_input, :]
                batch_out[iBatch] = prices[:, ix_date_start[iBatch]+n_days_input, -1]   # closing price of next day after range of input days
            batch_in = np.reshape(batch_in, (1, n_symbols, n_days_input*4))
            batch_out = np.reshape(batch_out, (1, n_symbols))

            # If any price for a symbol is nan, remove that symbol
            bKeep = np.logical_not(np.any(np.isnan(batch_in), axis=2))[0]
            batch_in = batch_in[:,bKeep,:]
            batch_out = batch_out[:,bKeep]

            bKeep = np.logical_not(np.any(np.isnan(batch_out), axis=0))
            batch_in = batch_in[:,bKeep,:]
            batch_out = batch_out[:,bKeep]

            # Convert to Variable and move to GPU if GPU available
            batch_in = V(TT(batch_in.astype(np.float32)))
            batch_out = V(TT(batch_out.astype(np.float32)))

            # Put data through model and compute loss
            output = net(batch_in)
            loss = criterion(output, batch_out)

            # Use negative profit as the loss. Not sure the right way to normalize this to "unity neutral" investment.
            output = output / torch.norm(output, p=2, dim=1)
            output = output - torch.mean(output)
            loss_per_stock = torch.exp(batch_out) * output[:,:,0] - output[:,:,0]
            loss_day = np.append(loss_day,(-torch.sum(loss_per_stock)).data.cpu().numpy())    # profit = SUM[gi*xi - xi], gi=linear gain, xi=investment (negative if short)

        # day_mean_loss = 1-np.prod(1-loss_day)**(1.0/len(loss_day))
        # loss_test_history = np.append(loss_test_history,day_mean_loss)
        day_median_loss = np.median(loss_day)
        loss_test_history = np.append(loss_test_history,day_median_loss)


plt.figure(1)
plt.clf()

# plt.subplot(2,1,1)
# plt.semilogy(loss_history,'.')
# v = plt.axis()
# plt.semilogy([v[0], v[1]], np.mean(loss_baseline)*np.ones(2))
# plt.title('loss baseline = %0.5f' % (np.mean(loss_baseline)))
# plt.grid()

plt.subplot(2,1,1)
plt.plot(loss_history ,'.')
v = plt.axis()
plt.plot([v[0], v[1]], np.zeros(2))
plt.title('Training sample loss')
plt.grid()

plt.subplot(2,1,2)
plt.plot(loss_test_history, '-o')
v = plt.axis()
plt.plot([v[0], v[1]], [0, 0])
plt.title('Test set median daily loss')
plt.grid()

# plt.subplot(2,1,2)
# plt.plot(spear,'.')
# v = plt.axis()
# plt.plot([v[0], v[1]], [0, 0])
# plt.title('Spearman (r)')
# plt.grid()


plt.figure(2)
plt.clf()
no = output[0,:,0].data.cpu().numpy()
bo = batch_out[0,:].data.cpu().numpy()
plt.plot(bo,no,'o');
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.grid()

sys.exit()
# Get weights
weights = []
grads = []
for param in net.parameters():
    weights.append(param.data)
    grads.append(param.grad)


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
from torch.autograd import Variable as V
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.ion()

## Set model parameters
layer_sizes = [3, 4096]  # (conv1, fc1). Excluding final output layer.
dropout = 0.5
batch_size = 64
n_iters = 1000
n_days_input = 10

################################################################
# Initialize some things, and define functions and classes
################################################################

fn = 1  # figure number

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

b_use_cuda = torch.cuda.is_available()

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
        self.fc2 = nn.Linear(n_symbols*output_sizes[0], output_sizes[1], bias=True)
        self.do2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(output_sizes[1], output_sizes[2], bias=True)

    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = output.view(-1, self.n_symbols * self.n_kernels)
        output = self.do1(output)
        output = F.relu(self.fc2(output))
        output = self.do2(output)
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


# Remove stocks with low mean volume
# TODO: Remove bad stocks based on additional factors
mean_vol = np.nanmean(volume, axis=1)
thresh = np.median(mean_vol)
ix = np.where(mean_vol>thresh)
symbols = symbols[ix]
volume = volume[ix]
prices = prices[ix]

# TODO: To help cope with outliers that make model training difficult, try just labeling
# prices changes as up or down. And have model just pick the amount to place on each stock
# rather than predicting up or down. Only allow fixed total amount to be placed across all
# sticks.


# Use price gains relative to previous close, not absolute dollars
prices[:, 1:, :] = prices[:, 1:, :] / prices[:, 0:-1, 3:4]
prices[:, 0:1, :] = prices[:, 0:1, :] / prices[:, 0:1, 0:1]   # Relative to open price, for first day of data

# Replace all zeros and +/-infs by nans
prices[prices==0] = np.nan
volume[volume==0] = np.nan
prices[np.isinf(prices)] = np.nan
volume[np.isinf(volume)] = np.nan

# Just you binary up/down to deal with outliers and bad data
ix = np.where(np.isnan(prices))
prices = (prices>1)*2.0-1
prices[ix] = np.nan

# # Use log-scale prices. How to deal with zeros?
# prices = np.log10(prices)
# prices[np.isinf(prices)] = np.nan

# Normalize prices and/or volume?
# TODO: This is not valid for cross validation.
#prices = (prices - np.mean(prices)) / np.std(prices)

# Center dayofweek about zero
dayofweek = np.asarray(dayofweek, dtype=np.float32) - 2.0


# Use last N days as hold-out for testing
n_symbols = len(symbols)
n_dates = len(dates)
n_ho_days = 26*5

prices_ho = prices[:,-n_ho_days:,:]
volume_ho = volume[:,-n_ho_days:]
dates_ho = dates[-n_ho_days:]
dayofweek_ho = dayofweek[-n_ho_days:]

prices = prices[:,:-n_ho_days,:]
volume = volume[:,:-n_ho_days]
dates = dates[:-n_ho_days]
dayofweek = dayofweek[:-n_ho_days]

n_symbols = len(symbols)
n_dates = len(dates)


# Build model
n_symbols = len(symbols)
n_dates = len(dates)
layer_sizes = layer_sizes + [n_symbols]  # adding element to list, not adding value to elements
n_features = n_days_input * 4   # 4 is because we have prices for open, high, low, close
net = fc_layers(n_symbols, n_features, layer_sizes, dropout=dropout)
if b_use_cuda:
    net = net.cuda()


# Train model
optimizer = optim.Adam(net.parameters())

loss_history = np.zeros(n_iters)
# loss_train_history = np.asarray([])
# loss_test_history = np.asarray([])

criterion = nn.MSELoss()

plt.figure(fn)
fn += 1

n_iter_per_log = 10
t_start = time.time()
for iter in range(n_iters):
    net.train()

    # Initialize for batch/iteration
    optimizer.zero_grad()

    # Build the batch
    ix_date_start = np.random.randint(0, n_dates-n_days_input, batch_size)
    batch_in = np.zeros((batch_size, n_symbols, n_days_input, 4))
    batch_out = np.zeros((batch_size, n_symbols))
    for iBatch in range(batch_size):
        batch_in[iBatch] = prices[:, ix_date_start[iBatch]:ix_date_start[iBatch]+n_days_input, :]
        batch_out[iBatch] = prices[:, ix_date_start[iBatch]+n_days_input, -1]   # closing price of next day after range of input days
    batch_in = np.reshape(batch_in, (batch_size, n_symbols, n_days_input*4))
    batch_out = np.reshape(batch_out, (batch_size, n_symbols))

    # If any price for a symbol is nan, set all prices to zero...
    bUse = np.logical_not(np.isnan(batch_out))
    mask = np.zeros_like(bUse, dtype=np.float32)
    mask[bUse] = 1.0
    iZero = np.where(np.isnan(batch_in))
    batch_in[iZero[0],iZero[1],:] = 0
    batch_out[iZero[0],iZero[1]] = 0
    iZero = np.where(np.isnan(batch_out))
    batch_in[iZero[0],iZero[1],:] = 0
    batch_out[iZero[0],iZero[1]] = 0


    # Shuffle the symbol ordering ...
    # TODO: Might want to use different shuffling for each sample in the batch
    ix_shuffle = np.arange(batch_size)
    np.random.shuffle(ix_shuffle)
    batch_in = batch_in[ix_shuffle]
    batch_out = batch_out[ix_shuffle]

    # Convert to Variable and move to GPU if GPU available
    if b_use_cuda:
        batch_in = V(torch.from_numpy(batch_in.astype(np.float32)).contiguous().cuda())
        batch_out = V(torch.from_numpy(batch_out.astype(np.float32)).contiguous().cuda())
        mask = V(torch.from_numpy(mask.astype(np.float32)).contiguous().cuda())
    else:
        batch_in = V(torch.from_numpy(batch_in.astype(np.float32)).contiguous())
        batch_out = V(torch.from_numpy(batch_out.astype(np.float32)).contiguous())
        mask = V(torch.from_numpy(mask.astype(np.float32)).contiguous())

    # Put data through model and compute loss
    output = net(batch_in)
    loss = criterion(output, batch_out)
    # loss = criterion(torch.mul(output, mask), torch.mul(batch_out, mask))
    #loss = - torch.mul(torch.sum(torch.mul(torch.mul(output, batch_out-1.0), mask)), 1./np.sum(bUse))    # amount of money made (negative) per trade
    #loss = - torch.mul(torch.sum(torch.mul(torch.mul(output, batch_out-1.0), mask)), 1./np.sum(bUse))    # amount of money made (negative) per trade
    # TODO: Add heavy L1 cost to promote small number of trades?



    # Compute gradients and update parameters
    loss.backward()
    optimizer.step()

    loss_history[iter] = loss.data.cpu().numpy()

    # if iter==40:
    #     plt.figure(40)
    #     plt.imshow(output.data.numpy(), aspect='auto')
    #     plt.colorbar()
    #     plt.figure(41)
    #     plt.imshow(batch_out.data.numpy(), aspect='auto')
    #     plt.colorbar()
    # if iter==41:
    #     plt.figure(51)
    #     plt.imshow(output.data.numpy(), aspect='auto')
    #     plt.colorbar()
    #     plt.figure(52)
    #     plt.imshow(batch_out.data.numpy(), aspect='auto')
    #     plt.colorbar()

    if (iter+1)%n_iter_per_log==0:
        print('Iteration %d, Loss=%0.3f, Duration=%0.3f' % (iter+1, loss.data.cpu().numpy(), time.time()-t_start))

        plt.cla()
        plt.plot(loss_history)
        plt.grid()
        plt.show()
        plt.pause(0.1)

        # ## Put full continuous training and testing data through model
        # net.eval()

        # encoder_hidden = net.initHidden(1)
        # encoder_output, _ = net(data_train.unsqueeze(2).transpose(2,1), encoder_hidden)
        # loss = criterion(encoder_output[:,:,0:n_labels].contiguous().view(1*data_train.size()[0],-1), \
        #              labels_train.transpose(1,0).contiguous().view(1*data_train.size()[0])) # cross-entropy
        # loss_train_history = np.append(loss_train_history,loss.data.cpu().numpy())

        # encoder_hidden = net.initHidden(1)
        # encoder_output, _ = net(data_test.unsqueeze(2).transpose(2,1), encoder_hidden)
        # loss = criterion(encoder_output[:,:,0:n_labels].contiguous().view(1*data_test.size()[0],-1), \
        #              labels_test.transpose(1,0).contiguous().view(1*data_test.size()[0])) # cross-entropy
        # loss_test_history = np.append(loss_test_history,loss.data.cpu().numpy())

'''
plt.figure(fn)
fn += 1
plt.clf()
plt.imshow(batch_out.data.cpu().numpy(), aspect='auto')
plt.colorbar()
plt.title('Batch ground truth')

plt.figure(fn)
fn += 1
plt.clf()
plt.imshow(output.data.cpu().numpy(), aspect='auto')
plt.colorbar()
plt.title('Batch prediction')

plt.figure(fn)
fn += 1
i = 0
plt.clf()
plt.subplot(1,2,1)
plt.plot(batch_out[i].data.cpu().numpy().flatten(), output[i].data.cpu().numpy().flatten(), '.')
plt.xlabel('Ground truth')
plt.ylabel('Prediction')
plt.grid()
plt.subplot(1,2,2)
n = np.arange(n_symbols)
plt.plot(n, batch_out[i].data.cpu().numpy().flatten(), n, output[i].data.cpu().numpy().flatten())
plt.grid()


net.eval()
output = net(batch_in)


plt.figure(fn)
fn += 1
plt.clf()
plt.imshow(output.data.cpu().numpy(), aspect='auto')
plt.colorbar()
plt.title('Batch prediction')

plt.figure(fn)
fn += 1
i = 0
plt.clf()
plt.subplot(1,2,1)
plt.plot(batch_out[i].data.cpu().numpy().flatten(), output[i].data.cpu().numpy().flatten(), '.')
plt.xlabel('Ground truth')
plt.ylabel('Prediction')
plt.grid()
plt.subplot(1,2,2)
n = np.arange(n_symbols)
plt.plot(n, batch_out[i].data.cpu().numpy().flatten(), n, output[i].data.cpu().numpy().flatten())
plt.grid()
'''

## Test on hold-out data
output_all = np.zeros((n_ho_days-n_days_input-1, n_symbols))
batch_out_all = np.zeros((n_ho_days-n_days_input-1, n_symbols))
for i in range(n_ho_days-n_days_input-1):
    batch_in = prices_ho[:, i:i+n_days_input, :]
    batch_out = prices_ho[:, i+n_days_input, -1]
    batch_in = np.reshape(batch_in, (1, n_symbols, n_days_input*4))
    batch_out = np.reshape(batch_out, (1, n_symbols))

    # If any price for a symbol is nan, set all prices to zero...
    bUse = np.logical_not(np.isnan(batch_out))
    mask = np.zeros_like(bUse, dtype=np.float32)
    mask[bUse] = 1.0
    iZero = np.where(np.isnan(batch_in))
    batch_in[iZero[0],iZero[1],:] = 0
    batch_out[iZero[0],iZero[1]] = 0
    iZero = np.where(np.isnan(batch_out))
    batch_in[iZero[0],iZero[1],:] = 0
    batch_out[iZero[0],iZero[1]] = 0

    # # Shuffle the symbol ordering ...
    # # TODO: Might want to use different shuffling for each sample in the batch
    # ix_shuffle = np.arange(batch_size)
    # np.random.shuffle(ix_shuffle)
    # batch_in = batch_in[ix_shuffle]
    # batch_out = batch_out[ix_shuffle]

    # Convert to Variable and move to GPU if GPU available
    if b_use_cuda:
        batch_in = V(torch.from_numpy(batch_in.astype(np.float32)).contiguous().cuda())
        batch_out = V(torch.from_numpy(batch_out.astype(np.float32)).contiguous().cuda())
        mask = V(torch.from_numpy(mask.astype(np.float32)).contiguous().cuda())
    else:s
        batch_in = V(torch.from_numpy(batch_in.astype(np.float32)).contiguous())
        batch_out = V(torch.from_numpy(batch_out.astype(np.float32)).contiguous())
        mask = V(torch.from_numpy(mask.astype(np.float32)).contiguous())

    # Put data through model and compute loss
    output = net(batch_in)
    loss = criterion(output, batch_out)
    output_all[i,:] = output.data.cpu().numpy()
    batch_out_all[i,:] = batch_out.data.cpu().numpy()


plt.figure(fn)
fn += 1
plt.clf()
plt.plot(batch_out_all.flatten(), output_all.flatten(), '.')
plt.xlabel('Ground truth')
plt.ylabel('Prediction')
plt.grid()

plt.figure(fn)
fn += 1
i = 0
plt.clf()
n = np.arange(n_symbols)
plt.plot(n,batch_out_all[i,:], n,output_all[i,:])
plt.xlabel('Ground truth')
plt.ylabel('Prediction')
plt.grid()


# Get weights
weights = []
for param in net.parameters():
    weights.append(param.data)


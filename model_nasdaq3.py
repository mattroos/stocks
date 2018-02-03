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


# TODO: Outputs are all identical ofter only one iteration. Something wrong.
# Getting weights with nans after 1 iteration. Bad gradients, somehow.

# TODO: How to use larger batch size?

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

plt.ion()

## Set model parameters
layer_sizes = [20, 20, 1]
dropout = 0.0
batch_size = 1
n_iters = 2000
n_days_input = 10
n_iter_per_log = 100
learn_rate = 0.0003

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
        output_summ = self.LinearSumm1(output/output.size()[2])   # divided by number of stocks
        output_summ = torch.sum(output_summ, 1, keepdim=True)   # sum over all stocks
        output_summ = F.relu(output_summ)
        output_summ = output_summ + self.LinearSumm2(output_summ)
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
    prices = np.log10(prices)
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
# loss_train_history = np.asarray([])
# loss_test_history = np.asarray([])
loss_baseline = np.zeros(n_iters)

criterion = nn.MSELoss()

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
    #loss = criterion(torch.mul(output[:,:,0],mask), torch.mul(batch_out,mask))
    #loss = - torch.mul(torch.sum(torch.mul(torch.mul(output, batch_out-1.0), mask)), 1./np.sum(bUse))    # amount of money made (negative) per trade
    # TODO: Add heavy L1 cost to promote small number of trades?

    # Compute gradients and update parameters
    loss.backward()
    optimizer.step()

    loss_history[iter] = loss.data.cpu().numpy()
    loss_baseline[iter] = criterion(torch.zeros(batch_out.size()), batch_out).data.cpu().numpy()


    if (iter+1)%n_iter_per_log==0:
        print('Iteration %d, Loss=%0.5f, Duration=%0.3f' % (iter+1, loss.data.cpu().numpy(), time.time()-t_start))


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


plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.semilogy(loss_history,'.')
v = plt.axis()
plt.semilogy([v[0], v[1]], np.mean(loss_baseline)*np.ones(2))
plt.title('loss baseline = %0.5f' % (np.mean(loss_baseline)))
plt.grid()

plt.subplot(2,1,2)
no = output[0,:,0].data.cpu().numpy()
bo = batch_out[0,:].data.cpu().numpy()
r = spearmanr(no,bo)[0]
plt.plot(bo,no,'o');
plt.title('Spearman = %0.3f' % (r))
plt.xlabel('Truth')
plt.xlabel('Prediction')
plt.grid()

sys.exit()
# Get weights
weights = []
grads = []
for param in net.parameters():
    weights.append(param.data)
    grads.append(param.grad)


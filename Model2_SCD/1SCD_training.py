# %%
import torch
import numpy as np
import normflows as nf

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
# sys.path.append(os.path.join(os.path.dirname(c_directory), 'FCYeast'))

from tqdm import tqdm
import architecture
import SCD_simulator
#import eZplot
from SCD_plot import SCD_plot

from matplotlib import pyplot as plt
from tqdm import tqdm


enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCD_simulator.adjust_device(device)


# %%
# Define flows
model_file = 'network.pt'
figs_direc = 'network_perform'

model = architecture.make_model(context_size=2,tail_bound=15)

# %%
# Define target
SCD_simulator.adjust_device(device)
target = SCD_simulator.target()

# %%
try:
    model.load_state_dict(torch.load(model_file))
    print('loading pretrained network')
except:
    print('starting from scratch')

# %%
max_iter   = 2000
show_iter  = 100
n_batch    = 16

x = target.sample(N=1024*1024)
batch_size = x.size(0)//n_batch

x,context = x[:,0].reshape(-1,1)*1.0,x[:,1:]

batches = torch.arange(x.size(0)).reshape(n_batch,-1)

# %%
# Train model
loss_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

# %%
for it in tqdm(range(max_iter)):
    loss_epoch = np.array([])

    for it2 in (range(n_batch)):  
        optimizer.zero_grad() 
        # Compute loss
        batch = batches[it2]
        loss = -model.log_prob(x[batch], context[batch]).mean()
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
        loss_epoch = np.append(loss_epoch, loss.to('cpu').item())

    # Log loss
    loss_hist = np.append(loss_hist, np.mean(loss_epoch))

    if (it+1)%show_iter==0:
        index = (loss_hist.size+np.arange(-int(2.5*show_iter),0))

        
        with torch.no_grad():
            if len(loss_hist) >= show_iter:
                if loss_hist[-1] < loss_hist[-show_iter]:
                    torch.save(model.state_dict(), model_file)
                else:
                    #if the model got worse, go back to last and resample
                    model.load_state_dict(torch.load(model_file))
            else:
                torch.save(model.state_dict(), model_file)
        SCD_plot(model,target,loss_hist,index[index>=0])

    else:
        with torch.no_grad():
            samples_new = target.sample(N=16*1024)

            x[:samples_new.size(0)] = samples_new[:,0].reshape((-1,1))
            context[:samples_new.size(0)] = samples_new[:,1:]

            shuffle_index = torch.randperm(x.size(0))
            x = x[shuffle_index]
            context = context[shuffle_index]

# %%
torch.save(model.state_dict(), model_file)
np.savetxt(model_file.split('.')[0]+'_loss_hist.csv',loss_hist)


# %%




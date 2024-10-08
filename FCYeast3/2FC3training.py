# %%
import torch
import numpy as np
import normflows as nf

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
sys.path.append(os.path.join(os.path.dirname(c_directory), 'FCYeast'))

from tqdm import tqdm
import architecture
import FCYeast_simulator
import eZplot

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
dil= sys.argv[1]
dil_factor = int(dil)/100
folder = 'dilution'+dil
model_file = folder+'/FCYeast_network.pt'
figs_direc = folder+'/FCYeast_network_perform'

# %%
# Define target
FCYeast_simulator.adjust_device(device)
target = FCYeast_simulator.target()


target.prior.loc[0]-=torch.log(torch.tensor(dil_factor,device=device))
target.params_dist.loc = target.prior.loc


# %%
model = architecture.make_model()

# %%
try:
    model.load_state_dict(torch.load(model_file))
    print('loading pretrained network')
except:
    print('starting from scratch')

# %%
max_iter   = 1500
show_iter  = 100
n_batch    = 16


x = target.sample(n=1024*1024)
batch_size = x.size(0)//n_batch

x,context = x[:,0].reshape(-1,1)*1.0,x[:,1:]

batches = torch.arange(x.size(0)).reshape(n_batch,-1)

# %%
# Train model
loss_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

print('starting training', dil,target.prior.loc.cpu(),context.mean(axis=0))
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
        
            loss_epoch = np.append(loss_epoch, 1.0*loss.to('cpu').item())
        del loss #this is a test

    # Log loss
    loss_hist = np.append(loss_hist, np.mean(loss_epoch))

    if (it+1)%show_iter==0:
        index = (loss_hist.size+np.arange(-int(2.5*show_iter),0))
        eZplot.presenting_results(model,target,loss_hist,index[index>=0],figs_direc = figs_direc)

    if (it + 1) % (3*show_iter) == 0:
        with torch.no_grad():
            torch.save(model.state_dict(), model_file)

    with torch.no_grad():
        samples_new = target.sample(n=64*1024)

        x[:samples_new.size(0)] = samples_new[:,0].reshape((-1,1))
        context[:samples_new.size(0)] = samples_new[:,1:]

        shuffle_index = torch.randperm(x.size(0))
        x = (x[shuffle_index]).contiguous()
        context = (context[shuffle_index]).contiguous()


# %%
torch.save(model.state_dict(), model_file)

# %%




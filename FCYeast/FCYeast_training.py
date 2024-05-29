# %%
import torch
import numpy as np
import normflows as nf

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))

from tqdm import tqdm
import architecture
import FCYeast_simulator
import eZplot

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
FCYeast_simulator.adjust_device(device)
target = FCYeast_simulator.target()

model = architecture.make_model(device=device)
model_file = 'FCYeast_network.pt'

# %%
try:
    model.load_state_dict(torch.load(model_file))
    #model.eval()  # Set the model to evaluation mode
    print('loading pretrained network')
except:
    print('starting from scratch')

# %%
max_iter   = 4000
show_iter  = 100
n_batch    = 16

x = target.sample(n=1024*1024)
batch_size = x.size(0)//n_batch

x,context = x[:,0].reshape(-1,1)*1.0,x[:,1:]

batches = torch.arange(x.size(0)).reshape(n_batch,-1)

# %%

loss_hist = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

# %%
for it in tqdm(range(max_iter)):
    loss_epoch = []

    for it2 in (range(n_batch)):  
        optimizer.zero_grad() 
        
        # Compute loss
        batch = batches[it2]
        loss = -model.log_prob(x[batch], context[batch]).mean()
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
            loss_epoch.append(1.0*loss.to('cpu').item())

        del loss

    loss_hist.append(np.mean(loss_epoch))

    if (it+1)%show_iter==0:
        eZplot.presenting_results(model,target,loss_hist)

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



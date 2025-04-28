# %%
import torch
import numpy as np
import normflows as nf
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
import architecture

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
def make_model_graph(model):
    x = model.sample(1000)[0]
    x = torch.linspace(x.min().item()-1,x.max().item()+1,1001).to(x.device)
    lpx = model.log_prob(x.reshape(-1,1)).detach()
    px = torch.exp(lpx)


    plt.plot(x.cpu(),px.cpu(),color='k',label='Normalizing flow')
    plt.hist(data.cpu().reshape(-1),bins=40,density=True,label='Intensity at higher dilution')
    plt.draw()

    plt.xlabel(r'$\log_{10} I$ (autofluorescence)')
    plt.ylabel("Density")
    plt.legend()
    plt.savefig('autofluo.png',dpi=600)
    

    plt.close()


# %%
data = pd.read_csv('../clean_data/complete_d=0.33.csv')['FL1-A'].to_numpy().astype(np.float32)
data = torch.log(torch.tensor(data).reshape(-1,1)).to(device)


# %%
model = architecture.make_model(conditional=False)
model.q0.loc += data.mean().item() 
model.q0.log_scale += torch.log(data.std()).item()


# %%
loss_hist = []
optimizer = torch.optim.Adam(model.parameters(), lr=1/data.numel())

# %%
for it in tqdm(range(250)):
    optimizer.zero_grad()
    loss = -model.log_prob(data).mean()
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        
        loss_hist.append( 1.0*loss.to('cpu').item())

    del loss



# %%
#plt.ioff()
make_model_graph(model)

torch.save(model.state_dict(), 'autofluorescence.pt')





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
import DCD_simulator
import eZplot

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from matplotlib import pyplot as plt

# %%

def make_plot(ax,th,model,target):
    #values = th.cpu().numpy()
    N=2**15
    x = target.sample(th,N=N,return_lparams=False)
    # ax.set_title(r'$\Psi_{{\beta}}$ = {:.2f} $\Psi_{{\lambda_{{act}}}}$ = {:.2f} $\Psi_{{\lambda_{{ina}}}}$  = {:.2f} $\Psi_{{\sigma}}$  = {:.2f}  '.format(*values))
    
    xx = torch.linspace(x.min()*.95,x.max()*1.05,201,device=th.device) 
    ly = model.log_prob(xx.reshape(-1,1), th.repeat(xx.size(0),1)).detach()
    
    x,xx,ly = x.cpu(),xx.cpu(),ly.cpu()

    dx= (xx[1]-xx[0])
    y = torch.exp(ly-ly.max())
    y = y/(y.sum()*dx)

    ax.plot(xx,y,label='NN likelihood')
    ax.hist(x.reshape((1,-1)),density=True,bins=35,label='Simulation')
    
    ax.legend()

def make_figure(model,target,loss_hist):
    fig, ax = plt.subplots(1,3,figsize=(9,4))
    ths = (target.lbeta_dist.loc -1,target.lbeta_dist.loc,target.lbeta_dist.loc+1)
    for (th,axi) in zip(ths,ax.reshape(-1)):
        make_plot(axi,th,model,target)
    #make_plot(ax[0],target.lbeta_dist.loc,model,target)
    return fig
    

# %%
model_file = 'network.pt'
figs_direc = 'network_perform'

# %%
model = architecture.make_model(context_size=1,tail_bound=10)

# %%
DCD_simulator.adjust_device(device)
target = DCD_simulator.target()

# %%
# max_iter   = 20000
# show_iter  = 500

max_iter   = 2000
show_iter  = 100

n_batch    = 16

x = target.sample(N=1024*1024)
x,context = x[:,0].reshape(-1,1)*1.0,x[:,1:]

batch_size = x.size(0)//n_batch
batches = torch.arange(x.size(0)).reshape(n_batch,-1)

# %%
# Train model
loss_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

# %%
for it in tqdm(range(max_iter)):
    loss_epoch = np.array([])

    for it2 in range(n_batch):  
        optimizer.zero_grad() 
        
        batch = batches[it2]
        loss = -model.log_prob(x[batch], context[batch]).mean()
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            loss_epoch = np.append(loss_epoch, 1.0*loss.item())
            
        del loss 

    # Log loss
    loss_hist = np.append(loss_hist, np.mean(loss_epoch))

    if (it+1)%show_iter==0:
        index = (loss_hist.size+np.arange(-int(2.5*show_iter),0))
        with torch.no_grad():
            if len(loss_hist) >= show_iter:
                if loss_hist[-1] < loss_hist[-show_iter]:
                    torch.save(model.state_dict(), model_file)
                else:
                    model.load_state_dict(torch.load(model_file))
            else:
                torch.save(model.state_dict(), model_file)
                
    #    eZplot.presenting_results(model,target,loss_hist,index[index>=0],figs_direc = figs_direc)
        make_figure(model,target,loss_hist) 
        epoch = len(loss_hist)
        plt.savefig(figs_direc+'/epoch{:>5}.png'.format(epoch),dpi=600)
        plt.close()   

    else:
        with torch.no_grad():
            samples_new = target.sample(N=16*1024)

            x[:samples_new.size(0)] = samples_new[:,0].reshape((-1,1))
            context[:samples_new.size(0)] = samples_new[:,1:]

            shuffle_index = torch.randperm(x.size(0))
            x = (x[shuffle_index]).contiguous()
            context = (context[shuffle_index]).contiguous()

# %%
torch.save(model.state_dict(), model_file)

# %%
make_figure(model,target,loss_hist) 

# %%
loss_hist

# %%
np.savetxt('loss_hist.csv',loss_hist)

# %%




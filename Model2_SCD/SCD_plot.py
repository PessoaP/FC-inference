import torch
import numpy as np
from matplotlib import pyplot as plt

def SCD_make_plot(ax,th,model,target):
    #x = target.sample(torch.exp(th[0]),torch.exp(th[1]),n=2**15,return_lparams=False)
    x = target.sample(th[0],th[1],N=2**15,return_lparams=False)
    xx = torch.linspace(x.min()*.9,x.max()*1.1,201,device=x.device) 
    ly = model.log_prob(xx.reshape(-1,1), th.repeat(xx.size(0),1)).detach()
    
    x,xx,ly = x.cpu(),xx.cpu(),ly.cpu()

    dx= (xx[1]-xx[0])
    y = torch.exp(ly-ly.max())
    y = y/(y.sum()*dx)

    ax.plot(xx,y,label='NN likelihood')
    ax.hist(x.reshape((1,-1)),density=True,bins=40,label='Simulation')
    ax.set_title(r'$\Psi_\beta$ = {:.2f}  $\Psi_\sigma$ = {:.2f}'.format(*(th.cpu().numpy())))
    ax.legend()
    #plt.show()

def SCD_plots_graph(ax,model,target):
    model.eval()
    mu = torch.stack((target.lbeta_dist.mean, target.lsig_dist.mean))
    sig= torch.stack((target.lbeta_dist.scale,target.lsig_dist.scale))

    th = mu + sig * torch.tensor(((-3, 0),
                                  ( 0, 0),
                                  ( 3, 0),
                                  (-3,-3),
                                  ( 3, 3)),device = sig.device)
    
    for axi,thi in zip(ax.reshape(-1),th):
        SCD_make_plot(axi,thi,model,target)
    
    model.train()

def SCD_plot(model,target,loss_hist=None,index = None,figs_direc ='network_perform'):
        fig, ax = plt.subplots(2,3,figsize=(18,8))
        SCD_plots_graph(ax,model,target)
        
        ax[-1,-1].plot(index,loss_hist[index])
        ax[-1,-1].set_ylabel('loss')
        #ax[-1,-1].set_ylabel('loss')
        #plt.show()

        [axi[0].set_ylabel('density') for axi in ax]

        epoch = index[-1]+1
        fig.suptitle('Epoch: {:>5}'.format(epoch),fontsize=15)
        fig.tight_layout()

        plt.savefig(figs_direc+'/epoch{:>5}.png'.format(epoch),dpi=600)
        #plt.clf()
        plt.close()
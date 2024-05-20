import torch
import numpy as np
from matplotlib import pyplot as plt
from eZsamplers import random_binary

def make_plot(ax,th,model,target,system):
    values = th.cpu().numpy()
    n=2**15
    if system == 'BSCD':
        x = target.sample(th[:2], th[2:4], th[4],n=n,return_lparams=False)
        ax.set_title(r'$\Psi_{{\beta_l}}$ = {:.2f}  $\Psi_{{\beta_h}}$ = {:.2f} $\Psi_{{\lambda_{{act}}}}$ = {:.2f} $\Psi_{{\lambda_{{ina}}}}$  = {:.2f} $\Psi_{{\sigma}}$  = {:.2f}'.format(*values))
    elif system == 'FCBSCD':      
        th_s = th*torch.ones((n//8,6),device=th.device)
        x= [target.sample(th_s[:,:2],th_s[:,2:4], 
                          th_s[:,4], th_s[:,5],n=n//8,return_lparams=False).cpu() for i in range(8)]
        x = torch.concatenate(x)
        ax.set_title(r'$\Psi_{{\beta_l}}$ = {:.2f}  $\Psi_{{\beta_h}}$ = {:.2f} $\Psi_{{\lambda_{{act}}}}$ = {:.2f} $\Psi_{{\lambda_{{ina}}}}$  = {:.2f} $\Psi_{{\sigma}}$  = {:.2f} $\Psi_{{\xi}}$  = {:.2f}'.format(*values),fontsize=8)
    
    xx = torch.linspace(x.min()*.95,x.max()*1.05,201,device=th.device) 
    ly = model.log_prob(xx.reshape(-1,1), th.repeat(xx.size(0),1)).detach()
    
    x,xx,ly = x.cpu(),xx.cpu(),ly.cpu()

    dx= (xx[1]-xx[0])
    y = torch.exp(ly-ly.max())
    y = y/(y.sum()*dx)

    ax.plot(xx,y,label='NN likelihood')
    ax.hist(x.reshape((1,-1)),density=True,bins=35,label='Simulation')
    
    ax.legend()

def plots_graph(ax,model,target,num_plots=6,system='BSCD'):
    model.eval()
    mu = target.params_dist.loc
    sig= (target.params_dist.covariance_matrix.diag())**(.5)

    th = mu + sig * (2. *random_binary(num_plots,len(mu)) *torch.sign(torch.randn(num_plots,len(mu))) ).to(mu.device) 
    
    for axi,thi in zip(ax.reshape(-1),th):
        make_plot(axi,thi,model,target,system)
    
    model.train()

def BSCD_plot(model,target,loss_hist=None,index = None,figs_direc = 'BSCD_network_perfom'):
        fig, ax = plt.subplots(3,3,figsize=(18,8))
        plots_graph(ax,model,target,8,system='BSCD')
        
        ax[-1,-1].plot(index,loss_hist[index])
        ax[-1,-1].set_ylabel('loss')
        ax[-1,-1].set_yscale('log')
        #ax[-1,-1].set_ylabel('loss')
        #plt.show()

        [axi[0].set_ylabel('density') for axi in ax]

        epoch = index[-1]+1
        fig.suptitle('Epoch: {:>5}'.format(epoch),fontsize=15)
        fig.tight_layout()

        plt.savefig(figs_direc+'/epoch{:>5}.png'.format(epoch),dpi=600)
        #plt.clf()
        plt.close()


def FCBSCD_plot(model,target,loss_hist=None,index = None,figs_direc = 'FCBSCD_network_perform'):
        fig, ax = plt.subplots(3,3,figsize=(18,8))
        plots_graph(ax,model,target,8,system='FCBSCD')
        
        ax[-1,-1].plot(index,loss_hist[index])
        ax[-1,-1].set_ylabel('loss')
        ax[-1,-1].set_yscale('log')

        [axi[0].set_ylabel('density') for axi in ax]

        epoch = index[-1]+1
        fig.suptitle('Epoch: {:>5}'.format(epoch),fontsize=15)
        fig.tight_layout()

        plt.savefig(figs_direc+'/epoch{:>5}.png'.format(epoch),dpi=600)
        #plt.clf()
        plt.close()
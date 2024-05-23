import torch

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
sys.path.append(os.path.join(os.path.dirname(c_directory), 'BSCD'))

import eZsamplers
import BSCD_simulator

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
BSCD_simulator.adjust_device(device)
eZsamplers.adjust_device(device)

mu = 1497
sig = 624
l0 = (mu/sig)**2
k = (sig**2)/mu


def adjust_device(dev):
    global device
    BSCD_simulator.adjust_device(dev)
    eZsamplers.adjust_device(dev)
    device = dev

def fix_data_type(lbeta,llam,lsig,lxi,n):
    #I assume either all are tensors, or all are single valued
    if torch.is_tensor(lbeta) and lbeta.ndim != 1: 
        if not(torch.is_tensor(lsig)) or lsig.ndim == 0:
            #if not tensor asume is a single value
            lbeta.to(device)
            llam.to(device)
            lsig=lsig*torch.ones((n,),device=device)
            lxi=lxi*torch.ones((n,),device=device)

        elif lbeta.shape != llam.shape:
            print('shapes do not match')
            return None
    else: #they are all single valued
        lbeta = lbeta*torch.ones((n,2),device=device)
        llam = llam*torch.ones((n,2),device=device)
        lsig = lsig*torch.ones((n,),device=device)
        lxi = lxi*torch.ones((n,),device=device)
    
    return lbeta,llam,lsig,lxi,lxi.size()[0]


ind=torch.arange(1024,device=device)
def adjust_indexes(n):
    global ind
    if (ind.dim != 1) or (ind.shape[0]!=n) :
        ind=torch.arange(n,device=device)
    
def simulator(beta,lam,sig,xi,rho,T=100,n=1024):
    #print('12',device,rho.x0.device)
    beta,lam,sig,xi,n = fix_data_type(beta,lam,sig,xi,n)
    adjust_indexes(n)
    t,x,s = BSCD_simulator.simulator(beta,lam,sig,rho,T,n)

    I = eZsamplers.ap_poisson(l0+xi*x)*k
    return t,I,s



class target():
    def __init__(self, means = (5.,8.,0.,0.,-2.3,0), sigmas=(1.,1.,1.,1.,.25,2)):
        self.prior = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))
        self.rho = eZsamplers.beta_sym(2.,6.,device=device)

    def change_sampling_distro(self, means = (5.,8.,0.,0.,-2.3,0), sigmas=(1.,1.,1.,1.,.25,2)):
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))

    def log_prior(self,x):
        return self.prior.log_prob(x)
        
    def sample(self, lbeta=None, llam=None, lsig=None, lxi=None,  T=100, n=1024,return_lparams=True):
        if lbeta == None:
            params = self.params_dist.sample((n,))

            beta = torch.exp(params[:,:2])
            lam  = torch.exp(params[:,2:4])
            sig  = torch.exp(params[:,4])
            xi  = torch.exp(params[:,5])

        else:
            params = torch.hstack((lbeta,llam,lsig.reshape(-1,1),lxi.reshape(-1,1)))*torch.ones((n,6),device=device)
            beta,lam,sig,xi = torch.exp(lbeta),torch.exp(llam),torch.exp(lsig),torch.exp(lxi)
            

 

        t,I,s = simulator(beta,lam,sig,xi,rho=self.rho,T=T,n=n)
        lI = torch.log(I.clamp(1.))

        if return_lparams:
            return torch.hstack((lI.reshape(-1,1),params))

        return lI
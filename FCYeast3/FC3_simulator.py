import torch

import sys
import os

c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
sys.path.append(os.path.join(os.path.dirname(c_directory), 'FCYeast'))

import eZsamplers
import FCYeast_simulator

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
FCYeast_simulator.adjust_device(device)
eZsamplers.adjust_device(device)

dils = torch.tensor([.12,.23,.33],device=device)
s_indices1 = torch.tensor([0, 1, 2, 3, 4, 11], device=device)
s_indices2 = torch.tensor([0, 1, 5, 6, 7, 11], device=device)
s_indices3 = torch.tensor([0, 1, 8, 9, 10, 11], device=device)

def adjust_device(dev):
    global device,dils,s_indices1,s_indices2,s_indices3,to_arbitrary
    FCYeast_simulator.adjust_device(dev)
    eZsamplers.adjust_device(dev)
    device = dev

    dils = torch.tensor([.12,.23,.33],device=device)
    s_indices1 = torch.tensor([0, 1, 2, 3, 4, 11], device=device)
    s_indices2 = torch.tensor([0, 1, 5, 6, 7, 11], device=device)
    s_indices3 = torch.tensor([0, 1, 8, 9, 10, 11], device=device)
    to_arbitrary = -torch.tensor([[[1, 1, 0, 0, 0, 0]],
                              [[1, 1, 0, 0, 0, 0]], 
                              [[1, 1, 0, 0, 0, 0]]], device=device) * torch.log(dils).reshape(-1, 1, 1).to(device)

    
def separate(x):
    group1 = x[:, s_indices1]
    group2 = x[:, s_indices2]
    group3 = x[:, s_indices3]

    # Stacking along a new dimension to separate the groups for each instance in the batch
    return torch.stack((group1, group2, group3), dim=0)

to_arbitrary = -torch.tensor([[[1, 1, 0, 0, 0, 0]],
                              [[1, 1, 0, 0, 0, 0]], 
                              [[1, 1, 0, 0, 0, 0]]], device=device) * torch.log(dils).reshape(-1, 1, 1).to(device)

def transform_to_arbitrary(x):  #suppose that \betas (0 and 1) are in hours, and the other in the arbirary units. Turn them all to arbitrary
    return separate(x) + to_arbitrary
    
ind=torch.arange(1024,device=device)
def adjust_indexes(n):
    global ind
    if (ind.dim != 1) or (ind.shape[0]!=n) :
        ind=torch.arange(n,device=device)

class target():
    def __init__(self,means = (1.5,5.,0.,0.,-2.3,0), sigmas=(1.,1.,1.,1.,.5,2)):
        self.t_base = FCYeast_simulator.target(means,sigmas)
        means = self.t_base.prior.loc[[0,1,2,3,4,2,3,4,2,3,4,5]] 
        sigmas = torch.sqrt(self.t_base.prior.covariance_matrix.diag())[[0,1,2,3,4,2,3,4,2,3,4,5]]

        self.prior = torch.distributions.MultivariateNormal(torch.tensor(means).clone().detach().to(device), torch.diag(torch.tensor(sigmas)**2).clone().detach().to(device))
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor(means).clone().detach().to(device), torch.diag(torch.tensor(sigmas)**2).clone().detach().to(device))
        self.rho = eZsamplers.beta_sym(2.,6.,device=device)

    #delete this in the future
    def sample(self, lbetas=None, llams=None, lsigs=None, lxis=None,  T=100, n=1024,return_lparams=True):
        if lbetas == None:
            params = self.params_dist.sample((n,))
            params_sep = transform_to_arbitrary(params)
                

            betas = torch.exp(params_sep[:,:,:2])
            lams  = torch.exp(params_sep[:,:,2:4])
            sigs  = torch.exp(params_sep[:,:,4])
            xis  = torch.exp(params_sep[:,:,5])


        else:
            betas,lams,sigs,xis = torch.exp(lbetas),torch.exp(llams),torch.exp(lsigs),torch.exp(lxis)

        lIs =[]
        for (beta,lam,sig,xi) in zip(betas,lams,sigs,xis):
            beta,lam,sig,xi,n = FCYeast_simulator.fix_data_type(beta[0],lam[0],sig[0],xi[0],n)

            t,I,s = FCYeast_simulator.simulator(beta,lam,sig,xi,rho=self.rho,T=T,n=n)

            lI = torch.log(I.clamp(1.))
            lIs.append(lI.reshape(-1,1))

        lIs = torch.hstack(lIs)

        return lIs

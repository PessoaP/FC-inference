import torch

import sys
import os

c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
import eZsamplers

from FCYeast import FCYeast_simulator

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

default_means = (torch.tensor(FCYeast_simulator.default_means)[[0,1,2,3,1,2,3]] ).to(device)
default_sigmas =(torch.tensor(FCYeast_simulator.default_sigmas)[[0,1,2,3,1,2,3]] ).to(device)

def adjust_device(dev):
    global device,dils,s_indices1,s_indices2,s_indices3,to_arbitrary
    FCYeast_simulator.adjust_device(dev)
    eZsamplers.adjust_device(dev)
    device = dev

    dils = torch.tensor([.12,.23],device=device)
    s_indices1 = torch.tensor([0, 1, 2, 3], device=device)
    s_indices2 = torch.tensor([0, 4, 5, 6], device=device)
    to_arbitrary = -torch.tensor([[1, 0, 0, 0],
                                  [1, 0, 0, 0]], device=device) * torch.log(dils).reshape(-1, 1).to(device)
adjust_device(device)
    
def separate(x):
    group1 = x[s_indices1]
    group2 = x[s_indices2]
    return torch.stack((group1, group2), dim=0)

def transform_to_arbitrary(x):  #suppose that \betas  are in hours, and the other in the arbirary units. Turn them all to arbitrary
    return separate(x) + to_arbitrary
    
ind=torch.arange(1024,device=device)
def adjust_indexes(n):
    global ind
    if (ind.dim != 1) or (ind.shape[0]!=n) :
        ind=torch.arange(n,device=device)

class target():
    def __init__(self, means = default_means,
                 sigmas=default_sigmas):
        self.prior = torch.distributions.MultivariateNormal((means).clone().detach().to(device), torch.diag((sigmas)**2).clone().detach().to(device))
        self.params_dist = torch.distributions.MultivariateNormal((means).clone().detach().to(device), torch.diag((sigmas)**2).clone().detach().to(device))
        self.rho = eZsamplers.beta_sym(6.,14.,device=device)


    def sample(self, lbetas=None, llams=None, lsigs=None,  T=100, N=1024,return_lparams=True):
        
        if lbetas == None: #no parameters provided
            params = self.params_dist.sample((N,))
            params_sep = transform_to_arbitrary(params)

            betas = torch.exp(params_sep[:,:1])
            lams  = torch.exp(params_sep[:,1:3])
            sigs  = torch.exp(params_sep[:,3:4])

        else:
            if llams == None: #assume parameters were provided together as first argument (lbetas)  
                    params = lbetas
                    betas = torch.exp(params[:,:1])
                    lams  = torch.exp(params[:,1:3])
                    sigs  = torch.exp(params[:,3:4])
            else:
                betas,lams,sigs = torch.exp(lbetas),torch.exp(llams),torch.exp(lsigs)

        lIs =[]
        #print( (beta.shape,lam.shape,sig.shape) )
        for (beta,lam,sig) in zip(betas,lams,sigs):
            beta,lam,sig,n = FCYeast_simulator.fix_data_type(beta,lam,sig,N)

            t,Pr,s = FCYeast_simulator.simulator(beta,lam,sig,rho=self.rho,T=T,N=n)
            I_prot = FCYeast_simulator.prot2intensity(Pr)

            del t,s

            lI = torch.logaddexp(FCYeast_simulator.autofluo.sample((N))[0],torch.log(I_prot))

            lIs.append(lI.reshape(-1,1))

        lIs = torch.hstack(lIs)

        return lIs

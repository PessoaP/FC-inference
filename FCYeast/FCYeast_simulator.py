import torch

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
sys.path.append(os.path.join(os.path.dirname(c_directory), 'BSCD'))

import eZsamplers
#import BSCD_simulator

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
#BSCD_simulator.adjust_device(device)
eZsamplers.adjust_device(device)

mu = 1497
sig = 624
l0 = (mu/sig)**2
k = (sig**2)/mu


def adjust_device(dev):
    global device
    #BSCD_simulator.adjust_device(dev)
    eZsamplers.adjust_device(dev)
    device = dev

def fix_data_type(lbeta,llam,lsig,lxi,n):    
    lbeta = lbeta*torch.ones((n,1),device=device)
    llam = llam*torch.ones((n,2),device=device)
    lsig = lsig*torch.ones((n,1),device=device)
    lxi = lxi*torch.ones((n,1),device=device)
    
    return lbeta,llam,lsig,lxi,lxi.size()[0]


def sample_initial(beta,lam,rho, n=1024):

    fraction_act = (lam[:,1]/lam.sum(dim=1)).reshape(-1,1)
    beta_eff = beta*fraction_act

    #Heuristically, it start at the steady-state of the non stochastic cell div.
    tau = torch.rand(beta_eff.shape,device=device)
    rate = beta_eff*(1+tau)
    x = eZsamplers.ap_poisson(rate)

    s = (torch.rand((n,1),device=device)<fraction_act).int()

    return tau,x,s

def simulate_between_cell_div(x,s,T,beta,lam,rho):  
    t = torch.zeros_like(T)#T means time until division (dt in the other func)  
    rate = torch.zeros_like(x)

    stop_changing= t>T

    while not(torch.all(stop_changing)):
        dt_prop = eZsamplers.exponential(lam[ind,s])#the first 0 is ridic please fix
        stop_changing = t + dt_prop > T

        dt_prop[stop_changing] = T[stop_changing]-t[stop_changing]
        t += dt_prop

        rate+=beta*dt_prop*(s==1).int()
        s = torch.where(stop_changing, s, 1 - s)

    x += eZsamplers.ap_poisson(rate)
    x = eZsamplers.ap_binomial(x,rho.sample(x.shape))
    return x,s

ind=torch.arange(1024,device=device).reshape(-1,1)
def adjust_indexes(n):
    global ind
    if (ind.dim != 1) or (ind.shape[0]!=n) :
        ind=torch.arange(n,device=device).reshape(-1,1)
    
def simulator(beta,lam,sig,xi,rho,T=100,n=1024):
    beta,lam,sig,xi,n = fix_data_type(beta,lam,sig,xi,n)
    div_time_dist = torch.distributions.LogNormal(0., sig)
    adjust_indexes(n)
    
    

    tau,x,s = sample_initial(beta,lam,rho,n)
    t = 1-tau
    x,s = simulate_between_cell_div(x,s,t,beta,lam,rho) #grow and divide for the time t-tau.

    T = T*torch.ones_like(x)
    dont_divide = t>T
    
    while not(torch.all(dont_divide)):
        #sample the next division time
        dt_prop = div_time_dist.sample()
        t_prop = t + dt_prop

        dont_divide = t_prop > T # indices where we already reach T.

        dt_prop[dont_divide] = T[dont_divide]-t[dont_divide] #the ones who overshot time only grow in the time between.
        t += dt_prop

        x_div,s_div = simulate_between_cell_div(x,s,dt_prop,beta,lam,rho)

        x_div[dont_divide] = x[dont_divide] # the ones who overshot time do not divide, we remove these
        s_div[dont_divide] = s[dont_divide]
        x = x_div
        s = s_div


    if torch.any(torch.isnan(x)):
        print('warning, it is returning nans')



    I = eZsamplers.ap_poisson(l0+xi*x)*k
    return t,I,s



class target():
    def __init__(self, means = (8.,0.,0.,-2.3,0), sigmas=(1.,1.,1.,.5,2)):
        self.prior = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))
        self.rho = eZsamplers.beta_sym(2.,6.,device=device)

    def log_prior(self,x):
        return self.prior.log_prob(x)
        
    def sample(self, lbeta=None, llam=None, lsig=None, lxi=None,  T=100, n=1024,return_lparams=True):
        if lbeta is None:
            params = self.params_dist.sample((n,))

            beta = torch.exp(params[:,:1])
            lam  = torch.exp(params[:,1:3])
            sig  = torch.exp(params[:,3:4])
            xi   = torch.exp(params[:,4:])

        else:
            lbeta,llam,lsig,lxi,void= fix_data_type(lbeta,llam,lsig,lxi,n)
            beta,lam,sig,xi = torch.exp(lbeta),torch.exp(llam),torch.exp(lsig),torch.exp(lxi)


        t,I,s = simulator(beta,lam,sig,xi,rho=self.rho,T=T,n=n)
        lI = torch.log(I.clamp(1.))

        if return_lparams:
            return torch.hstack((lI.reshape(-1,1),params))

        return lI
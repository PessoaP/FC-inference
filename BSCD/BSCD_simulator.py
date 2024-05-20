import torch

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
import eZsamplers

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
#device = torch.device('cpu')
eZsamplers.adjust_device(device)

def adjust_device(dev):
    global device
    eZsamplers.adjust_device(dev)
    device = dev

def fix_data_type(lbeta,llam,lsig,n):
    #I assume either all are tensors, or all are single valued
    if torch.is_tensor(lbeta) and lbeta.ndim != 1: 
        if not(torch.is_tensor(lsig)) or lsig.ndim == 0:
            #if not tensor asume is a single value
            lbeta.to(device)
            llam.to(device)
            lsig=lsig*torch.ones((n,),device=device)

        elif lbeta.shape != llam.shape:
            print('shapes do not match')
            return None
    else: #they are all single valued
        lbeta = lbeta*torch.ones((n,2),device=device)
        llam = llam*torch.ones((n,2),device=device)
        lsig = lsig*torch.ones((n,),device=device)
    
    return lbeta,llam,lsig,lsig.size()[0]

def sample_initial(beta,lam,rho, n=1024):
    if (torch.is_tensor(beta) and beta.ndim==1):
        beta =  beta*torch.ones((n,2)).to(device)
        lam =  lam*torch.ones((n,2)).to(device)

    lam_eff = lam/(lam.sum(dim=1).reshape(-1,1))
    beta_eff = (beta*lam_eff).sum(dim=1)
    #print(beta_eff)


    #Heuristically, it start at the steady-state of the non stochastic cell div.
    tau = torch.rand(beta_eff.shape,device=device)
    rate = beta_eff*(1+tau)
    x = eZsamplers.ap_poisson(rate)

    s = (torch.rand(n,device=device)<lam_eff[:,0]).int()

    return tau,x,s

ind=torch.arange(1024,device=device)
def adjust_indexes(n):
    global ind
    if (ind.dim != 1) or (ind.shape[0]!=n) :
        ind=torch.arange(n,device=device)
    
def simulate_between_cell_div(x,s,T,beta,lam,rho):  
    #T means time until division (dt in the other func)  
    t = torch.zeros_like(T)
    rate = torch.zeros_like(x)

    stop_changing= t>T

    while not(torch.all(stop_changing)):
        dt_prop = eZsamplers.exponential(lam[ind,s])#the first 0 is ridic please fix

        stop_changing = t + dt_prop > T

        dt_prop[stop_changing] = T[stop_changing]-t[stop_changing]
        t += dt_prop

        rate+=beta[ind,s]*dt_prop
        s = torch.where(stop_changing, s, 1 - s)

    x += eZsamplers.ap_poisson(rate)
    x = eZsamplers.ap_binomial(x,rho.sample(x.shape))
    return x,s

def simulator(beta,lam,sig,rho=None,T=100,n=1024):
    if rho is None:
        rho = eZsamplers.delta(.5,device=device)

    beta,lam,sig,n = fix_data_type(beta,lam,sig,n)
    div_time_dist = torch.distributions.LogNormal(0., sig) #dist from which we sample the next division.

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
        #print(t)

    resample = torch.isnan(x)
    #print('this should be zero', resample.sum())
    #print(beta[resample][0],lam[resample][0],sig[resample][0])

    if torch.any(resample):
        t[resample],x[resample],s[resample] = simulator(beta[resample],lam[resample],sig[resample],rho,T[resample]) 

        
    return t,x,s



class target():
    def __init__(self, means = (5.,8.,0.,0.,-2.3), sigmas=(1.,1.,1.,1.,.25)):
        self.prior = torch.distributions.MultivariateNormal(torch.tensor((5.,8.,0.,0.,-2.3)).to(device), torch.diag(torch.tensor((1.,1.,1.,1.,.25))**2).to(device))
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))

    def change_sampling_distro(self, means = (5.,8.,0.,0.,-2.3), sigmas=(1.,1.,1.,1.,.25)):
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))

    def log_prior(self,x):
        return self.prior.log_prob(x)
        
    def sample(self, lbeta=None, llam=None, lsig=None,  T=100, n=1024,return_lparams=True):
        if lbeta == None:
            params = self.params_dist.sample((n,))

            beta = torch.exp(params[:,:2])
            lam  = torch.exp(params[:,2:4])
            sig  = torch.exp(params[:,4])

        elif not (torch.is_tensor(lbeta) and torch.is_tensor(lsig)) or lbeta.ndim ==1:
            params = torch.hstack((lbeta,llam,lsig))*torch.ones((n,5),device=device)
            lbeta,llam,lsig,n = fix_data_type(lbeta,llam,lsig,n)
            beta,lam,sig = torch.exp(lbeta),torch.exp(llam),torch.exp(lsig)
 

        t,x,s = simulator(beta,lam,sig,T=T,n=n)
        lx = torch.log(x.clamp(1.))

        if return_lparams:
            return torch.hstack((lx.reshape(-1,1),params))

        return lx
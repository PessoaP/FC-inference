import torch
import eZsamplers

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
eZsamplers.adjust_device(device)

def adjust_device(dev):
    global device
    eZsamplers.adjust_device(dev)
    device = dev

def fix_data_type(beta,sig,N):
    #I assume either both are tensors, or both are single valued
    if torch.is_tensor(beta) and beta.ndim != 0:
        if not(torch.is_tensor(sig)) or sig.ndim == 0:
            #if not tensor asume is a single value
            beta.to(device)
            sig=sig*torch.ones_like(beta)

        elif beta.shape != sig.shape:
            print('shape of beta and sig tensors do not match')
            return None
    else:
        beta = beta*torch.ones((N,),device=device)
        sig = sig*torch.ones((N,),device=device)
    
    return beta,sig

def sample_initial(beta,rho=.5,N=1024):
    if isinstance(beta,float) or isinstance(beta,int) or (torch.is_tensor(beta) and beta.ndim==0):
        beta =  beta*torch.ones((N,)).to(device)

    #Heuristically, it start at the steady-state of the non stochastic cell div.
    tau = torch.rand(beta.shape,device=device)
    rate = beta*(1+tau)
    x = eZsamplers.ap_poisson(rate)

    return tau,x

def simulate_between_cell_div(x,dt,beta,rho=.5):    
    x += eZsamplers.ap_poisson(beta*dt)#counts increase poisson in between cell div
    #x =  eZsamplers.ap_binomial(x,rho*torch.ones_like(x))# divide all cells
    return x
def cell_divide(x,rho=.5):
    # Samples of rho are the volume ratio of cells before and after division, as such each protein will have a probability rho.sample of being in the cell aafter division
    return eZsamplers.ap_binomial(x,rho*torch.ones_like(x))


def simulator(beta,sig,rho=.5,T=100,N=1024):
    beta,sig = fix_data_type(beta,sig,N)
    div_time_dist = torch.distributions.LogNormal(0., sig) #dist from which we sample the next division.

    tau,x = sample_initial(beta,rho)
    t = 1-tau
    x = simulate_between_cell_div(x,t,beta,rho=.5) #grow and divide for the time t-tau.

    T = T*torch.ones_like(beta)
    dont_divide = t>T
    
    while not(torch.all(dont_divide)):
        #sample the next division time
        dt_prop = div_time_dist.sample()
        t_prop = t + dt_prop
        dont_divide = t_prop > T #inds are indices where we already reach T.

        dt_prop[dont_divide] = T[dont_divide]-t[dont_divide] #the ones who overshot time only grow in the time between.
        t += dt_prop

        x = simulate_between_cell_div(x,dt_prop,beta,rho)
        x = torch.where(dont_divide,x,cell_divide(x,rho)) #If the cell divides, reduce the number accordingly
       
    return x


class target():
    def __init__(self):
        self.lbeta_dist = torch.distributions.Normal(torch.tensor(7.).to(device),torch.tensor(1.).to(device))
        self.beta_sampler = lambda shape: torch.exp(self.lbeta_dist.sample(shape))

        self.lsig_dist = torch.distributions.Normal(torch.tensor(-2.3).to(device),torch.tensor(.25).to(device))
        self.sig_sampler = lambda shape: torch.exp(self.lsig_dist.sample(shape))

        #fix
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor((7.,-2.3)).to(device), torch.diag(torch.tensor((1,.25))**2).to(device))
        

    def log_prior(self,params):
        beta,sig = params[0],params[1]
        return self.lbeta_dist.log_prob(beta)+self.lsig_dist.log_prob(sig)
    
    def sample(self, lbeta=None, lsig=None, T=100, N=1024,return_lparams=True):
        if lbeta == None:
            lbeta = self.lbeta_dist.sample((N,))
            lsig = self.lsig_dist.sample((N,))

            # beta = torch.exp(lbeta)
            # sig = torch.exp(lsig)

        elif not (torch.is_tensor(lbeta)) or lbeta.ndim==0:
            lbeta,lsig = fix_data_type(lbeta,lsig,N)
            #beta,sig = torch.exp(lbeta),torch.exp(lsig)
 
        beta,sig = torch.exp(lbeta),torch.exp(lsig)
 
        x = simulator(beta,sig,T=T,N=N)

        lx = torch.log(x.clamp(1.))

        if return_lparams:
            return torch.stack((lx,lbeta,lsig),dim=1)

        return lx
import torch
import eZsamplers

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
eZsamplers.adjust_device(device)

def adjust_device(dev):
    global device
    eZsamplers.adjust_device(dev)
    device = dev

def fix_data_type(beta,n):
    if not(torch.is_tensor(beta)) or beta.ndim == 0:
        beta = beta*torch.ones((n,),device=device)
    return beta

def sample_initial(beta,rho=.5,N=1024):
    if isinstance(beta,float) or isinstance(beta,int) or (torch.is_tensor(beta) and beta.ndim==0):
        beta =  beta*torch.ones((N,)).to(device)
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


def simulator(beta,rho=.5,T=100,N=1024):
    beta = fix_data_type(beta,N)
    ### div_time_dist = torch.distributions.LogNormal(0., sig) #dist from which we sample the next division.

    tau,x = sample_initial(beta,rho)
    t = 1-tau
    x = simulate_between_cell_div(x,t,beta,rho=.5) #grow and divide for the time t-tau.

    T = T*torch.ones_like(beta)
    dont_divide = t>T
    
    while not(torch.all(dont_divide)):
        #sample the next division time
        dt_prop = torch.ones_like(beta)###div_time_dist.sample()
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

    def log_prior(self,beta):
        return self.lbeta_dist.log_prob(beta)
    

    def sample(self, lbeta=None, T=100, N=1024,return_lparams=True):
        if lbeta == None:
            lbeta = self.lbeta_dist.sample((N,))


        elif not (torch.is_tensor(lbeta)) or lbeta.ndim==0:
            lbeta = fix_data_type(lbeta,N)
        beta = torch.exp(lbeta)
 
        x = simulator(beta,N=N)

        lx = torch.log(x.clamp(1.))

        if return_lparams:
            return torch.stack((lx,lbeta),dim=1)

        return lx
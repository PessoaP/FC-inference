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


def simulator(beta,rho=.5,N=1024):
    
    beta = fix_data_type(beta,N)

    tau = torch.rand(beta.shape,device=device)
    rate = beta*(1+tau)
    x = eZsamplers.ap_poisson(rate)
        
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
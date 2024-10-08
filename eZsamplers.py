#easy samplers
import torch
import numpy as np

gaussian_base = torch.distributions.Normal(0.,1.)
expo_base = torch.distributions.Exponential(1.)


def ap_binomial(N,rho,size=1):
    if isinstance(N,np.ndarray) or isinstance(N,np.float64) or isinstance(N,np.float32) or isinstance(N,float) or isinstance(N,int):
        N = torch.tensor(N)
    if isinstance(size,int):
        size = (size,)
    if torch.is_tensor(N) and N.ndim==0:
        N = N*torch.ones(size)

    base = gaussian_base.sample(N.shape)
    mean = N*rho
    var = mean*(1-rho)
    res =  (mean+base*torch.sqrt(var))

    replace = torch.logical_and(mean<50,mean-3*torch.sqrt(var)<20)

    if torch.any(replace):
        bin_sampler = torch.distributions.Binomial((mean[replace]).round().int(),rho[replace])
        res[replace] = bin_sampler.sample()*1.0

    return res

def ap_poisson(rate,size=1):
    if isinstance(rate,np.ndarray) or isinstance(rate,np.float64) or isinstance(rate,np.float32) or isinstance(rate,float) or isinstance(rate,int):
        rate = torch.tensor(rate)
    if isinstance(size,int):
        size = (size,)
    if torch.is_tensor(rate) and rate.ndim==0:
        rate = rate*torch.ones(size)

    base = gaussian_base.sample(rate.shape)
    res = rate+base*torch.sqrt(rate)

    replace = rate<50
    if torch.any(replace):
        poisson_sampler = torch.distributions.Poisson(rate[replace])
        res[replace] = poisson_sampler.sample()*1.0

    return res

def gaussian_sample(mus,sigs):
    base = gaussian_base.sample(mus.shape)
    return mus+base*sigs

def exponential(lam,size=1,rate=True):
    if isinstance(lam,np.ndarray) or isinstance(lam,np.float64) or isinstance(lam,np.float32) or isinstance(lam,float) or isinstance(lam,int):
        lam = torch.tensor(lam)
    if isinstance(size,int):
        size = (size,)
    if torch.is_tensor(lam) and lam.ndim==0:
        lam = lam*torch.ones(size,device=expo_base.rate.device)

    base = expo_base.sample(lam.shape)

    if rate:
        return base/lam
    return base*lam

def int_to_binary(tensor, bits):
    powers_of_2 = 2**torch.arange(bits - 1, -1, -1).unsqueeze(1)
    binary_tensor = (tensor.unsqueeze(0) & powers_of_2).t() > 0

    return binary_tensor.int()

def random_binary(n, bits, zero_first = True):
    nums = torch.sort( torch.randperm(2**bits)[:n] )[0]
    if zero_first:
        nums[0] = 0
    return int_to_binary(nums, bits)



class delta():
    #makes a distribution that always sample x0
    def __init__(self, x0, device='cpu'):
        #super(delta,self).__init__()
        self.x0 = torch.tensor(x0).to(device)
    
    def sample(self,shape):
        return self.x0 + torch.zeros(shape,device=self.x0.device)
    
    def log_prob(self,x):
        return torch.where(x==self.x0, 0., -torch.inf)
    
    
class beta_sym():
    def __init__(self, alpha, beta, device='cpu'):
        self.alpha,self.beta = torch.tensor(alpha).to(device), torch.tensor(beta).to(device)
        self.dist = torch.distributions.Beta(self.alpha,self.beta)
    
    def sample(self,shape):
        x = self.dist.sample(shape)
        f = torch.rand(shape,device=self.alpha.device)<.5
        return torch.where(f, 1-x, x)
    
    def log_prob(self,x):
        lp = torch.stack((self.dist.log_prob(x),self.dist.log_prob(1-x)))
        return torch.logsumexp(lp+np.log(1/2),axis=0)
    
def adjust_device(dev):
    global gaussian_base,expo_base
    gaussian_base = torch.distributions.Normal(torch.tensor(0.).to(dev),torch.tensor(1.).to(dev))#.to(device)
    expo_base = torch.distributions.Exponential(torch.tensor(1.).to(dev))

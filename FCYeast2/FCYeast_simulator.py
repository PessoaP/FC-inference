import torch
import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
import architecture
import eZsamplers

enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

eZsamplers.adjust_device(device)

#Loads the distribution of autofluorescence
autofluo = architecture.make_model(conditional=False)
autofluo.load_state_dict(torch.load('autofluorescence.pt'))
for param in autofluo.parameters():
    param.requires_grad = False


#These functions are not very important, they are only necessary to guarantee the code works correctly on the GPU.
def adjust_device(dev):
    global device
    eZsamplers.adjust_device(dev)
    device = dev

def fix_data_type(lbeta,llam,lsig,n):    
    lbeta = lbeta*torch.ones((N,1),device=device)
    llam = llam*torch.ones((N,2),device=device)
    lsig = lsig*torch.ones((N,1),device=device)
    return lbeta,llam,lsig,lsig.size()[0]

ind=torch.arange(1024,device=device).reshape(-1,1)
def adjust_indexes(N):
    global ind
    if (ind.dim != 1) or (ind.shape[0]!=N) :
        ind=torch.arange(n,device=device).reshape(-1,1)


def sample_initial(beta,lam,rho, N=1024):
    #Here it samples an initial state.
    #Heuristically, we use the ratio equivalent to the fraction of time spent in active state of the non stochastic cell div.
    #This will be ``close enough'' not to bias the steady-state obtained.
    fraction_act = (lam[:,1]/lam.sum(dim=1)).reshape(-1,1)
    beta_eff = beta*fraction_act

    tau = torch.rand(beta_eff.shape,device=device)
    rate = beta_eff*(1+tau)
    x = eZsamplers.ap_poisson(rate)
    s = (torch.rand((N,1),device=device)<fraction_act).int()

    return tau,x,s

def simulate_between_cell_div(x,s,T,beta,lam,rho):  
    # Here we assume that each cell starts with `x` proteins and grows for a time `T``.
    # `T` means time until division or the simulation end time. (`dt`` in the `simulator`` function) 
    t = torch.zeros_like(T)  
    rate = torch.zeros_like(x)
    #stop_changing = t>T

    while not(torch.all(stop_changing)):
        dt_prop = eZsamplers.exponential(lam[ind,s]) #Samples the time until the next state switch
        stop_changing = t + dt_prop > T #Creates a mask for cells that are not going to switch because they will divide first.

        dt_prop[stop_changing] = T[stop_changing]-t[stop_changing]
        t += dt_prop #Moves time either to the next state switch or the next cell division, whichever is closer

        rate+=beta*dt_prop*((s==1).int()) # Adds to the rate only if the cell were in the active state
        s = torch.where(stop_changing, s, 1 - s) #If it switches states, switch state

    x += eZsamplers.ap_poisson(rate) #Since the sum of Poisson is Poisson with the sum of rates, it is enough to just sample the protein production once.
    return x,s

def cell_divide(x,rho):
    # Samples of rho are the volume ratio of cells before and after division, as such each protein will have a probability rho.sample of being in the cell aafter division
    return eZsamplers.ap_binomial(x,rho.sample(x.shape))
    
def simulator(beta,lam,sig,rho,T=100,N=1024):
    beta,lam,sig,n = fix_data_type(beta,lam,sig,N)
    div_time_dist = torch.distributions.LogNormal(0., sig)
    adjust_indexes(N)    

    tau,x,s = sample_initial(beta,lam,rho,N)
    t = 1-tau
    x,s = simulate_between_cell_div(x,s,t,beta,lam,rho) #grow and divide for the time t-tau.

    T = T*torch.ones_like(x)
    dont_divide = t>T
    
    while not(torch.all(dont_divide)):
        dt_prop = div_time_dist.sample() #Sample the time at which we have the next cell division
        t_prop = t + dt_prop
        dont_divide = t_prop > T # indices where we already reach T means the cell do not divide

        dt_prop[dont_divide] = T[dont_divide]-t[dont_divide] #The ones who overshot time only grow in the time between.
        t += dt_prop

        x,s = simulate_between_cell_div(x,s,dt_prop,beta,lam,rho)
        x = torch.where(dont_divide,x,cell_divide(x,rho)) #If the cell divides, reduce the number accordingly


    if torch.any(torch.isnan(x)):
        print('warning, the simulation is returning NaNs')

    return t,x,s

xi = 0.05 #Value obtained from 
def prot2intensity(Pr,xi=xi,tol=1e-3):
    return (xi*Pr + torch.sqrt(xi*Pr)*torch.randn_like(Pr)).clamp(xi*tol)

class target():
    def __init__(self, means = (10,-1.,1.,-2.3), sigmas=(3.,1.,1.,.75)):
        self.prior = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))
        self.params_dist = torch.distributions.MultivariateNormal(torch.tensor(means).to(device), torch.diag(torch.tensor(sigmas)**2).to(device))
        self.rho = eZsamplers.beta_sym(2.,6.,device=device)

    def log_prior(self,x):
        return self.prior.log_prob(x)
        
    def sample(self, lbeta=None, llam=None, lsig=None,  T=100, N=1024,return_lparams=True):
        #The sample parameters are 
        #1 - lbeta: log(base e) of the protein production rate,
        #2 - llam: 2value with the log(base e) activation and inactivation rates respectively, 
        #3 - lsig: log(base e) of the variance in cell division.
        # If you do not provide these values we will sample those from the parameters distribution and a different one for each sample
        # Not that we assume a unit of time that is the average cell division time (or inverse dilution in the chemostat). 
        #4 - T: The time at which we stop the simulation, we hope it reaches steady-state in the time needed for 100 cell divisions. 
        #5 - N: Number of samples 

        if lbeta is None:
            params = self.params_dist.sample((N,))
            beta = torch.exp(params[:,:1])
            lam  = torch.exp(params[:,1:3])
            sig  = torch.exp(params[:,3:4])

        else:
            lbeta,llam,lsig,void= fix_data_type(lbeta,llam,lsig,n)
            beta,lam,sig = torch.exp(lbeta),torch.exp(llam),torch.exp(lsig)

        t,Pr,s = simulator(beta,lam,sig,rho=self.rho,T=T,N=N) #Simulator returns the times, final number of protein and final state
        I_prot = prot2intensity(Pr) #We turn the number of protein into an intensity for protein only
        lI = torch.logaddexp(autofluo.sample((N))[0],torch.log(I_prot)) #Have the log of the total intensity by taking the log of a sample of autofluorescence plus the protein intensity.
        
        if return_lparams:
            return torch.hstack((lI.reshape(-1,1),params))

        return lI
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


default_means = (10,-1.,1.,-2.3)
default_sigmas = (3.,.5,.5,.75)

#Loads the distribution of autofluorescence
try:
    autofluo = architecture.make_model(conditional=False)
    autofluo.load_state_dict(torch.load('../Real/autofluorescence.pt'))
    for param in autofluo.parameters():
        param.requires_grad = False
except:
    print('Unable to load autofluorescence distribution')


#These functions are not very important, they are only necessary to guarantee the code works correctly on the GPU.
def adjust_device(dev):
    global device
    eZsamplers.adjust_device(dev)
    device = dev
    autofluo.to(device)

def fix_data_type(lbeta,llam,lsig,N):    
    lbeta = lbeta*torch.ones((N,1),device=device)
    llam = llam*torch.ones((N,2),device=device)
    lsig = lsig*torch.ones((N,1),device=device)
    return lbeta,llam,lsig,lsig.size()[0]



ind=torch.arange(1024,device=device).reshape(-1,1)
def adjust_indexes(N):
    global ind
    if (ind.dim != 1) or (ind.shape[0]!=N) :
        ind=torch.arange(N,device=device).reshape(-1,1)


def sample_initial(beta,lam,rho, N=1024):
    #Here it samples an initial state.
    #Heuristically, we use the ratio equivalent to the fraction of time spent in active state of the non stochastic cell div.
    #This will be ``close enough'' not to bias the steady-state obtained.
    
    fraction_act = (lam[:,1]/lam.sum(dim=1)).reshape(-1,1)
    
    #For SI, in case the beta has 2 states. 
    if beta.ndim == 1 or beta.shape[-1] == 1:
        beta_eff = beta * fraction_act
    else:
        beta_eff = (1 - fraction_act) * beta[:, [0]] + fraction_act * beta[:, [1]]

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
    stop_changing = t>T
    while not(torch.all(stop_changing)):
        ##
        g = lam[ind[:lam.shape[0]],s]
        ##
        dt_prop = eZsamplers.exponential(lam[ind[:lam.shape[0]],s]) #Samples the time until the next state switch
        stop_changing = t + dt_prop > T #Creates a mask for cells that are not going to switch because they will divide first.

        dt_prop[stop_changing] = T[stop_changing]-t[stop_changing]
        t += dt_prop #Moves time either to the next state switch or the next cell division, whichever is closer

        if beta.ndim == 1 or beta.shape[-1] == 1:
            rate += beta * dt_prop * (s == 1)
        else:
            rate += beta[ind, s] * dt_prop

        
        s = torch.where(stop_changing, s, 1 - s) #If it switches states, switch state

    x += eZsamplers.ap_poisson(rate) #Since the sum of Poisson is Poisson with the sum of rates, it is enough to just sample the protein production once.
    return x,s

def cell_divide(x,rho):
    # Samples of rho are the volume ratio of cells before and after division, as such each protein will have a probability rho.sample of being in the cell aafter division
    return eZsamplers.ap_binomial(x,rho.sample(x.shape))

def simulator(beta_mean,lam,sig,
              rho=eZsamplers.beta_sym(6.,14.,device=device),
              T=100,N=1024,
              z_evol_step = .01, beta_evol_step = .01):
    adjust_indexes(N) 
    
    beta_mean,lam,sig,N = fix_data_type(beta_mean,lam,sig,N)
    beta = beta_mean*(1+torch.randn(N,1,device=device)*.3).clamp(min=.1)
    nu = torch.pow(sig,-2)
    div_time_dist = torch.distributions.Gamma(nu,nu)

    tau,x,s = sample_initial(beta,lam,rho,N)
    Z = torch.ones_like(x) #div_time_dist.sample() 
    T_nextdiv = Z
    T_lastdiv = -tau*Z

    t = 0.
    dt =.01
    
    while t<T:
        # ---- Event times inside the step ----
        t_div_exact = T_lastdiv + T_nextdiv        # [N]
        tau_div     = t_div_exact - t              # [N]

        div_in_dt  = tau_div  <= dt
        nothing    = ~div_in_dt

        if torch.any(nothing):
            idx = nothing.reshape(-1)
            x[idx], s[idx] = simulate_between_cell_div(x[idx], s[idx], 
                                                       dt*torch.ones_like(x[idx]), 
                                                       beta[idx], lam[idx], 
                                                       rho)
            
        if torch.any(div_in_dt):
            idx = div_in_dt.reshape(-1)

            dt_before = tau_div[idx]
            dt_rem    = (dt - dt_before).clamp(min=0)

            # evolve to division time
            x[idx], s[idx] = simulate_between_cell_div(x[idx], s[idx], 
                                                       dt_before, 
                                                       beta[idx], lam[idx], 
                                                       rho)

            # split
            xm = cell_divide(x[idx], rho)
            xd = x[idx] - xm
            sm = s[idx].clone()
            sd = s[idx].clone()

            # remainder
            x_m_rem, s_m_rem = simulate_between_cell_div(
                xm, sm, dt_rem, beta[idx], lam[idx], rho
            )
            x_d_rem, s_d_rem = simulate_between_cell_div(
                xd, sd, dt_rem, beta[idx], lam[idx], rho
            )

            x[idx], s[idx] = x_m_rem, s_m_rem

            x_to_append = x_d_rem.clone()
            s_to_append = s_d_rem.clone()
            beta_to_append = (beta[idx]).clone() * ( 1 + (torch.randn_like(x_to_append)*beta_evol_step).clamp(min=-.3,max=.3))
            z_to_append = (Z[idx]).clone()

            T_lastdiv[idx] = t_div_exact[idx]
            div_time_dist = torch.distributions.Gamma(nu[idx],nu[idx])
            T_nextdiv[idx] = div_time_dist.sample()*Z[idx]

            if len(x_to_append)>0:
                idx_new = (torch.randperm(len(x)+len(x_to_append),device=device)[:N]).reshape(-1)

                x = torch.cat((x,x_to_append),dim=0)[idx_new].contiguous()
                s = torch.cat((s,s_to_append),dim=0)[idx_new].contiguous()
                beta = torch.cat((beta,beta_to_append),dim=0)[idx_new].contiguous()
                Z = torch.cat((Z,z_to_append),dim=0)[idx_new].contiguous()
                
                Z = Z/Z.mean()
                Z *= 1 + (torch.randn_like(Z)*z_evol_step).clamp(min=-.3,max=.3)

                T_lastdiv = torch.cat((T_lastdiv,t_div_exact[idx].clone()),dim=0)[idx_new].contiguous()
                T_nextdiv = torch.cat((T_nextdiv,T_nextdiv[idx].clone()),dim=0)[idx_new].contiguous().clamp(min=dt) 
        t+=dt

    return t,x,s, beta, Z

xi = 0.05 #Value obtained from 
def prot2intensity(Pr,xi=xi,tol=1e-3):
    return (xi*Pr + torch.sqrt(xi*Pr)*torch.randn_like(Pr)).clamp(xi*tol)


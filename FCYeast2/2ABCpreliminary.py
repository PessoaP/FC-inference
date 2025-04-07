# %%
import torch
import numpy as np
import normflows as nf
import pandas as pd
from tqdm import tqdm
import time

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
torch.no_grad()

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.join(c_directory, 'FCYeast2'))

from matplotlib import pyplot as plt
import FCYeast2_simulator
import architecture

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
initial_time=time.time()

# %%
dils_str = ['12','23']
dils = [.12,.23]
dfs = [pd.read_csv(os.path.dirname(c_directory) +'/clean_data/complete_d={}.csv'.format(d)) for d in dils]
x = [np.log(df['FL1-A'].to_numpy()) for df in dfs]

# %%
Nbins = 100

h,bins=[],[]
for xi in x:
    hi,bi = np.histogram(xi,bins=Nbins)
    h.append(hi)
    bins.append(torch.tensor(bi).to(device)) #bi will be compared to simulations, keep here


# %%
# default_means =  torch.tensor((10.,-1.,1. ,-2.3,-1.,1. ,-2.3),device=device)
# default_sigmas = torch.tensor(( 3.,1.5,1.5,1.0,1.5,1.5,1.0),device=device)
# target = FCYeast2_simulator.target(means=default_means,sigmas=default_sigmas)
default_means = torch.tensor( (10,0,0,-2.3,0,0,-2.3) ).to(device)
default_sigmas = torch.tensor( (3.,1,1,1,1,1,1) ).to(device)
target = FCYeast2_simulator.target(means=default_means,sigmas=default_sigmas)


def logprior(params):
    return target.prior.log_prob(params)

# %%
def histogram_gpu(data,bins):
    bins = bins.reshape(-1,1)
    arr1 = data > bins[:-1]
    arr2 = data <= bins[1:]
    arr = torch.logical_and(arr1,arr2)
    ans = arr.sum(axis=1)
    del arr1
    del arr2
    del arr
    return ans

# %%
eps = float(1/Nbins) #to prevent log of 0. sums to 1, as we had one extra datapoint spread into all bins
        
def ABC_log_likelihood(param,N=2**12):
    #simulate
    params_arbitrary = FCYeast2_simulator.transform_to_arbitrary(param)
    
    simulations = target.sample(params_arbitrary,N=N,return_lparams=False)
    ll = []
    for (hi,bi,si) in zip(h,bins,simulations.T):
        db = (bi[1]-bi[0]).item()
        counts = histogram_gpu(si,bins=bi).cpu().numpy() + eps
        pi = counts/(N+1) #turn into frequencies
        pi = pi/db #turn into pdf equivalent
        ll.append( (hi*np.log(pi)).sum() )
    return ll
               
def ABC_log_post(params,lprior=logprior):
    return sum([lp.sum() for lp in ABC_log_likelihood(params)]) + lprior(params)

# %%
params_1k = target.prior.sample((1000,))
print(params_1k.mean(axis=0))
best_param = target.prior.loc
lp_max = ABC_log_post(best_param)

for i in range(1,10,4):
    print(i)
    for par in (1/i)*(params_1k-target.prior.loc) + best_param:
        lp_par = ABC_log_post(par)
        if lp_par>=lp_max:
            best_param = par
            lp_max=lp_par
            print(best_param.cpu().numpy(),lp_max)
            
del params_1k

# %%
param = best_param
lp = lp_max

sampled_params = [param.cpu().numpy()]
sampled_logpost = [lp.item()]

# %%
S = (3.14)*torch.eye(7)*1e-4
S = torch.eye(7)*1e-3
mvn = torch.distributions.MultivariateNormal(torch.zeros(7,device=device),S.to(device))

def proposal(param):
    return param + mvn.sample()

# %%
for i in tqdm(range(10000)):
    lp = ABC_log_post(param) #resample approximation
    param_prop = proposal(param)
    lp_prop = ABC_log_post(param_prop)

    if np.log(np.random.rand(1))< (lp_prop-lp).item():
        param = param_prop
        lp = lp_prop

    sampled_params.append(param.cpu().numpy())
    sampled_logpost.append(lp.item())

    if i%100 == 99:
        print(i,param,lp, logprior(param))
        np.savetxt('FCYeast2_MCMC/ABC_results_{}.csv'.format(seed),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))
np.savetxt('FCYeast2_MCMC/ABC_results_{}.csv'.format(seed),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))

# %%
#Save results for easy acess
means = np.stack(sampled_params).mean(axis=0).round(decimals=1)
sigs =  (np.stack(sampled_params).std(axis=0)).clip(.5)
dictionary = {'training_means': means, 'training_sigmas':sigs}
[dictionary.update({'{}dilution'.format(dil): tensor}) for(dil,tensor) in zip(dils_str,FCYeast2_simulator.transform_to_arbitrary(torch.tensor(means).to(device)).cpu().numpy())]
torch.save(dictionary, 'ABC_estimates.pt')

print('Totaltime',time.time()-initial_time)



# %%
import torch
import numpy as np
import normflows as nf
from tqdm import tqdm

seed = 1337
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
try:
    perc = sys.argv[1]
    frac = int(perc)/100
except:
    frac = 1.
# %%

estimates = torch.load('ABC_estimates.pt',weights_only=False)
means_ABC = torch.tensor(estimates['training_means']).to(device)
sigmas_ABC = torch.tensor(estimates['training_sigmas']).to(device)
target = FCYeast2_simulator.target(means_ABC,sigmas_ABC)
print(means_ABC,sigmas_ABC)

# %%
dils_str = ['12','23']
dils = [.12,.23]
models=[]

for dil in dils_str:
    model = architecture.make_model()
    model_file = 'dilution{}/network.pt'.format(dil)
    model.load_state_dict(torch.load(model_file))
    
    for param in model.parameters():
        param.requires_grad = False
    models.append(model)

# %%
import pandas as pd
dfs = [pd.read_csv(os.path.dirname(c_directory) +'/clean_data/complete_d={}.csv'.format(d)) for d in dils]
x_full = [torch.tensor(df['FL1-A'].to_numpy().astype(np.float32)).reshape(-1,1).to(device) for df in dfs]

N = (min([xi.size(0) for xi in x_full]))
ind = torch.randperm(N)

x = []
for xi in x_full:
    Ni = xi.size(0)
    indi = torch.cat((ind.to(xi.device), torch.arange(N, Ni, device=xi.device))) <= Ni * frac
    x.append(torch.log(xi[indi]))
N=int(N*frac)


def logprior(params):
    if torch.any(torch.abs(params-means_ABC)/sigmas_ABC > 3):
        return -torch.inf
    return target.prior.log_prob(params) 

vectorize_params = [torch.ones(xi.size(0),4).to(device) for xi in x]
def log_likelihood(data,params,models):
    global vectorize_params
    lp = []
    for (data_sub,params_sub,vec_params_sub,model_sub) in zip(data,params,vectorize_params,models):
        if data_sub.size !=  vec_params_sub.size(0):
            vec_params_sub = torch.ones((data_sub.size(0),4),device=device)
        lp.append(model_sub.log_prob(data_sub,params_sub*vec_params_sub))

    return lp

def log_post(data,params,models,lprior=logprior):
    return sum([lp.sum() for lp in log_likelihood(data,FCYeast2_simulator.transform_to_arbitrary(params),models)]) + lprior(params)

# %%
params_1k = target.prior.sample((1000,))
best_param = target.prior.loc
lp_max = log_post(x,best_param,models)

for i in range(1,11):
    print(i)
    for par in ( (1/i)*(params_1k-target.prior.loc) + best_param ):
        lp_par = log_post(x,par,models)
        if lp_par>=lp_max:
            best_param = par
            lp_max=lp_par
            print(best_param,lp_max)
            
del params_1k

# %%
param = best_param
lp = lp_max

sampled_params = [param.cpu()]
sampled_logpost = [lp.item()]

# %%
S = (3.14)*torch.eye(7)*1e-4
mvn = torch.distributions.MultivariateNormal(torch.zeros(7,device=device),S.to(device))
def change_S(newS):
    global S
    global mvn

    S = newS
    mvn = torch.distributions.MultivariateNormal(torch.zeros(7,device=device),S.to(device))
def proposal(param):
    return param + mvn.sample()

# %%
count_of_safe=0
loopruns = 0

while count_of_safe <=10:
    for i in range(150):
        param_prop = proposal(param)
        lp_prop = log_post(x,param_prop,models)

        if torch.log(torch.rand(1))< (lp_prop-lp).item():
            param = param_prop
            lp = lp_prop

        sampled_params.append(param.cpu())
        sampled_logpost.append(lp.cpu().item())

    print(param,lp)

    acc_rate = np.mean([(sampled_params[i] - sampled_params[i-1]).sum().item()!=0 for i in range(-1,-101,-1)])

    if acc_rate>.2 and acc_rate<.5:
        count_of_safe += 1
    else:
        count_of_safe = 0

    if loopruns%3==2:
        change_S( (torch.stack(sampled_params[-200:]).T.cov() + torch.eye(7)*1e-8) * (2.4**2/(7)) )
    loopruns+=1

    print(loopruns,acc_rate, ' ', sampled_logpost[-1])#, 'priordist', (((best_param - target.prior.loc)/ target.prior.covariance_matrix.diag()).max()))
    
# %%
burnin = len(sampled_logpost)
for i in tqdm(range(100000)):
    param_prop = proposal(param)
    lp_prop = log_post(x,param_prop,models)

    if torch.log(torch.rand(1))< (lp_prop-lp).item():
        param = param_prop
        lp = lp_prop

    sampled_params.append(param.cpu().numpy())
    sampled_logpost.append(lp.cpu().item())

    if i%100 == 99:
        print(i,param,lp, logprior(param))
        np.savetxt('FCYeast2_MCMC/mcmc_real_results_{}datapoints.csv'.format(N),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))
        
# %%
np.savetxt('FCYeast2_MCMC/mcmc_real_results_{}datapoints.csv'.format(N),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))


# %%
import torch
import numpy as np
import normflows as nf

seed = 42
torch.manual_seed(seed)
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


# %%
import FCYeast.FCYeast_simulator 
target_after = FCYeast.FCYeast_simulator.target()

def grid_plot(x,param,model,log10 = False):
    xp = torch.linspace(x.min(),x.max(),101).to(device)
    lp = model.log_prob(xp.reshape(-1,1),torch.ones((101,4),device=device)*param)
    p = torch.exp(lp-lp.max())
    p *= 1/(p.sum()*(xp[1]-xp[0]))
    if log10:
        l10 = 2.302585
        return xp.cpu()/l10,p.cpu()*l10
    return xp.cpu(),p.cpu()

# %%
target = FCYeast2_simulator.target()

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

dils = [.12,.23]
dfs = [pd.read_csv(os.path.dirname(c_directory) +'/clean_data/complete_d={}.csv'.format(d)) for d in dils]
x = [torch.tensor(df['FL1-A'].to_numpy().astype(np.float32)).reshape(-1,1).to(device) for df in dfs]

#uncomment to use only 20% of the data
#N = 10*(min([xi.size(0) for xi in x])//10)
#ind = torch.arange(0,N)%10<2 

x = [torch.log(xi[:N][ind]) for xi in x]
N = ind.sum()

# %%
l,s = target.prior.loc, torch.sqrt(target.prior.covariance_matrix.diag())
zf = lambda params: (params-l)/s
def logprior(params):
    z = zf(params)
    if torch.any(torch.abs(z)>3.01):
        return - torch.inf
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
    for par in (1/i)*(params_1k-best_param) + best_param:
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
sampled_logpost = [lp.cpu().item()]

# %%
S = torch.eye(7)*1e-4
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

while count_of_safe <=15:
    for i in range(150):
        param_prop = proposal(param)
        lp_prop = log_post(x,param_prop,models)

        if torch.log(torch.rand(1))< (lp_prop-lp).item():
            param = param_prop
            lp = lp_prop

        sampled_params.append(param.cpu())
        sampled_logpost.append(lp.cpu().item())

        #print(param_prop)

    print(param,lp)

    acc_rate = np.mean([(sampled_params[i] - sampled_params[i-1]).sum().item()!=0 for i in range(-1,-101,-1)])

    if acc_rate>.2 and acc_rate<.5:
        count_of_safe += 1
    else:
        count_of_safe = 0

    if loopruns%3==2:
        change_S( (torch.stack(sampled_params[-200:]).T.cov() + torch.eye(7)*1e-8) * (2.4**2/(7)) )
        #sanity(best_param = 1.0*param)
    loopruns+=1

    print(loopruns,acc_rate, ' ', sampled_logpost[-1])#, 'priordist', (((best_param - target.prior.loc)/ target.prior.covariance_matrix.diag()).max()))
    
# %%
burnin = len(sampled_logpost)
for i in range(100000):
    param_prop = proposal(param)
    lp_prop = log_post(x,param_prop,models)

    if torch.log(torch.rand(1))< (lp_prop-lp).item():
        param = param_prop
        lp = lp_prop

    sampled_params.append(param.cpu().numpy())
    sampled_logpost.append(lp.cpu().item())

    if i%100 == 99:
        print(i,param,lp, logprior(param))
        np.savetxt('FCYeast2_MCMC/mcmc_real_results_{}seed_{}datapoints.csv'.format(seed,N),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))
        #sanity(best_param = torch.tensor(sampled_params[np.argmax(sampled_logpost)]))

# %%
np.savetxt('FCYeast2_MCMC/mcmc_real_results_{}seedd_{}datapoints.csv'.format(seed,N),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))


#%%
#Just to print the sanity checck
def sanity(best_param):
    l10 = 2.302585

    fig,ax = plt.subplots(1,3,figsize=(12,4))
    h = [axi.hist(xi.cpu().numpy().reshape(-1)/l10 ,density=True,bins=45,label='Real') for (axi,xi) in zip(ax,x)]
    
    params = FCYeast2_simulator.transform_to_arbitrary(best_param.to(device))
    [axi.plot(*grid_plot(xi,parami,model,log10=True),label='NF Density') for (xi,parami,axi,model) in zip(x,params,ax,models)]
    [ax[i].hist(target_after.sample(params[i,:1],params[i,1:3],params[i,3:4],n=2**16,return_lparams=False).cpu().numpy()/l10 ,
            density=True,
            bins=h[i][1],alpha=.5,label='MAP simulation')for i in range(2)]

    [axi.set_xlabel(r'$\log_{10}$ I') for axi in ax]

    [axi.set_title(r'Dilution {}$/h$'.format(d)) for  (axi,d) in zip(ax,[.12,.23])]
    [axi.set_xlim(.9*xi.cpu().min()/l10,xi.cpu().max()/l10) for  (axi,xi) in zip(ax,x)]

    ax[0].legend(loc=8,ncol=3,bbox_to_anchor=(.5,1.201))
    plt.tight_layout()
    plt.savefig('FCYeast2_MCMC/sanity_check_{}seed.png'.format(seed),dpi=600)

sanity(best_param)


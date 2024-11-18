# %%
import torch
import numpy as np
import normflows as nf

data_seed = 16
seed = 123
torch.manual_seed(seed)
torch.no_grad()
N=2**10

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))
sys.path.append(os.path.join(os.path.dirname(c_directory), 'FCYeast'))
import FCYeast2_simulator
import architecture

from matplotlib import pyplot as plt
enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
target = FCYeast2_simulator.target()
context_size = 4

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
X = np.loadtxt('FC3_synth/synth_{}.csv'.format(data_seed)).astype(np.float32)
x = [torch.tensor(X[:,i]).reshape(-1,1).to(device)[:N] for i in range(len(dils))]

# %%
def logprior(params):
    return target.prior.log_prob(params)

vectorize_params = [torch.ones(xi.size(0),5).to(device) for xi in x]

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
gt = np.loadtxt('FC3_synth/gt_map.csv').astype(np.float32)
gt = torch.tensor(gt[gt[:,0]==data_seed][0,1:],device=device)
gt

# %%
log_post(x,gt,models)

# %%
params_1000 = target.prior.sample((1000,))
best_param = params_1000[0]
lp_max = log_post(x,best_param,models)

for par in params_1000[1:]:
    lp_par = log_post(x,par,models)
    if lp_par>lp_max:
        best_param = par
        lp_max=lp_par
        print(best_param,lp_max)

for i in range(3,8):
    print(i)
    for par in (1/i)*(params_1000-best_param) + best_param:
        lp_par = log_post(x,par,models)
        if lp_par>lp_max:
            best_param = par
            lp_max=lp_par
            print(best_param,lp_max)


del params_1000

# %%
param = best_param
lp = lp_max

sampled_params = [param.cpu()]
sampled_logpost = [lp.cpu().item()]

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

    acc_rate = np.mean([(sampled_params[i] - sampled_params[i-1]).sum().item()!=0 for i in range(-1,-101,-1)])

    if acc_rate>.2 and acc_rate<.5:
        count_of_safe += 1
    else:
        count_of_safe = 0

    if loopruns%3==2:
        change_S( (torch.stack(sampled_params[-200:]).T.cov() + torch.eye(10)*1e-8) * (2.4**2/(10)) )
    loopruns+=1

    print(loopruns,acc_rate, '    ', sampled_logpost[-1])
    



# %%
burnin = len(sampled_logpost)
for i in range(100000):
    param_prop = proposal(param)
    lp_prop = log_post(x,param_prop,models)

    if torch.log(torch.rand(1))< (lp_prop-lp).item():
        param = param_prop
        lp = lp_prop

    sampled_params.append(param.cpu())
    sampled_logpost.append(lp.cpu().item())

    if i%100 == 99:
        print(i,param,lp)
        #print(i,param,lp)

# %%
np.savetxt('FC3_mcmc/results_{}seed_{}data_{}dp.csv'.format(seed,data_seed,N),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))

# %%
plt.plot(sampled_logpost)
plt.axhline(log_post(x,gt,models).cpu().numpy(),color='y')

# %%
#this still does a single n, please fix

def grid_plot(x,param,model):
    xp = torch.linspace(x.min(),x.max(),101).to(device)
    lp = model.log_prob(xp.reshape(-1,1),torch.ones((101,4),device=device)*param)
    p = torch.exp(lp-lp.max())
    p *= 1/(p.sum()*(xp[1]-xp[0]))
    return xp.cpu(),p.cpu()

# %%
#x_all = X = np.loadtxt('FCYeast3_synth/synth_{}.csv'.format(data_seed)).astype(np.float32)
x_all = [torch.tensor(X[:,i]).reshape(-1,1).to(device)[:-1] for i in range(3)]
fig,ax = plt.subplots(1,3,figsize=(12,4))
[axi.hist(xi.cpu().numpy().reshape(-1) ,density=True,bins=35) for (axi,xi) in zip(ax,x_all)]
params = FCYeast2_simulator.transform_to_arbitrary(best_param)
[axi.plot(*grid_plot(xi,parami,model)) for (xi,parami,axi) in zip(x_all,params,ax)]

x

plt.legend()

# %%
'FC3_mcmc/results_{}seed_{}data_{}dp.csv'.format(seed,data_seed,N),

# %%




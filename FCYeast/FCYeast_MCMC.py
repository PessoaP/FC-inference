# %%
import torch
import numpy as np
import normflows as nf

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))

import architecture
import FCYeast_simulator

from matplotlib import pyplot as plt

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
N = 2**10

seed_data = 42
seed_mcmc = 13
torch.manual_seed(seed_mcmc)

# %%
FCYeast_simulator.adjust_device(device)
target = FCYeast_simulator.target()

model = architecture.make_model(device=device)
model_file = 'FCYeast_network.pt'

# %%
for param in model.parameters():
    param.requires_grad = False

# %%
try:
    model.load_state_dict(torch.load(model_file))
    #model.eval()  # Set the model to evaluation mode
    print('loading pretrained network')
except:
    print('starting from scratch')

# %%
dv = np.loadtxt('FCYeast_synth/gt_map.csv')
gt = torch.tensor( dv[ dv[:,0]==seed_data ][0,1:] , device=device,dtype=torch.float)

x = np.loadtxt('FCYeast_synth/synth_{}.csv'.format(seed_data))[:N]
x = torch.tensor(x,device=device,dtype=torch.float).reshape(-1,1)
gt

# %%
plt.hist(x.reshape(-1).cpu(), density=True,bins=45)

xp = torch.linspace(x.min(),x.max(),101).to(device)
lp = model.log_prob(xp.reshape(-1,1),torch.ones((101,5),device=device)*gt)
p = torch.exp(lp-lp.max())
p *= 1/(p.sum()*(xp[1]-xp[0]))
plt.plot(xp.cpu(),p.detach().cpu())


# %%
vectorize_params = torch.ones(1024,5).to(device)

def log_likelihood(data,params,model):
    global vectorize_params
    if data.size !=  vectorize_params.size(0):
        vectorize_params = torch.ones((data.size(0),5),device=device)

    return model.log_prob(data,params*vectorize_params)

def log_post(data,params,model,lprior):
    return log_likelihood(data,params,model).sum() + lprior(params)


# %%
lp_gt = log_post(x,gt,model,target.log_prior)
print('ground truth log posterior:', lp_gt.item())

# %%
#first 100 from prior
params_100 = target.sample(n=100)[:,1:]
best_param = params_100[0]
lp_max = log_post(x,best_param,model,target.log_prior)

for par in params_100[1:]:
    lp_par = log_post(x,par,model,target.log_prior)
    if lp_par>lp_max:
        best_param = par
        lp_max=lp_par

for i in range(3,8):

    params_100 = (target.sample(n=100)[:,1:] - best_param)/i +best_param

    for par in params_100:
        lp_par = log_post(x,par,model,target.log_prior)
        if lp_par>lp_max:
            best_param = par
            lp_max=lp_par
            print(best_param,lp_max)


del params_100

# %%
param = best_param
lp = lp_max

sampled_params = [param.cpu()]
sampled_logpost = [lp.cpu().item()]

# %%
S = torch.eye(5)*1e-4
mvn = torch.distributions.MultivariateNormal(torch.zeros(5,device=device),S.to(device))

def change_S(newS):
    global S
    global mvn

    S = newS
    mvn = torch.distributions.MultivariateNormal(torch.zeros(5,device=device),S.to(device))

def proposal(param):
    return param + mvn.sample()

# %%
count_of_safe=0
loopruns = 0

while count_of_safe <=10:
    for i in range(150):
        param_prop = proposal(param)
        lp_prop = log_post(x,param_prop,model,target.log_prior)

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
        change_S( (torch.stack(sampled_params[-200:]).T.cov() + torch.eye(5)*1e-10) * ((2.4**2)/5)) 
    loopruns+=1

    print(loopruns,acc_rate,lp)
    



# %%
sampled_params

# %%
param

# %%
plt.plot(sampled_logpost)
plt.axhline(lp_gt.item(),color='y')

# %%
burnin = len(sampled_logpost)
for i in range(100000):
    param_prop = proposal(param)
    lp_prop = log_post(x,param_prop,model,target.log_prior)

    if torch.log(torch.rand(1))< (lp_prop-lp).item():
        param = param_prop
        lp = lp_prop

    sampled_params.append(param.cpu())
    sampled_logpost.append(lp.cpu().item())

    if i%100 == 99:
        print(i,param,lp)
        #print(i,param,lp)

# %%
np.savetxt('FCYeast_mcmc/results_{}seed_{}data_{}dp.csv'.format(seed_mcmc,seed_data,N),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))

# %%
fig, ax = plt.subplots(2,3,figsize=(16,8))
labels = [r'$\log \beta$',r'$\log {\lambda_{act}}$',r'$\log {\lambda_{ina}}$',r'$\log \sigma$',r'$\log k$']
sp = torch.stack(sampled_params).numpy()

for (i,axi) in zip(range(5),ax.reshape(-1)):
    his = axi.hist(sp[burnin:burnin+sp.shape[0]//2,i],density=True,bins=25)
    #axi.hist(sp[burnin+sp.shape[0]//2:,i],density=True,bins=his[1],alpha=.5)
    #axi.hist(sp[burnin:,i],density=True,bins=25)
    axi.axvline(gt.cpu()[i].item(),color='k')
    axi.set_xlabel(labels[i],fontsize=14)

ax[-1][-1].plot(sampled_logpost)
ax[-1][-1].axvline(burnin,color='red')
ax[-1][-1].axhline(lp_gt.item(),color='y')

ax[-1][-1].set_ylabel('log posterior')

plt.suptitle('{} datapoints -- seed {}'.format(N,seed_mcmc))

plt.tight_layout()
plt.savefig('FCYeast_mcmc/fig_posterior/mcmc_noncentral_{}_{}data_{}dp.png'.format(seed_mcmc,seed_data,N),dpi=600)

# %%
map_param = sampled_params[np.argmax(sampled_logpost)].to(device)
map_param

# %%
fig, ax = plt.subplots(1)
ax.hist(x.reshape(-1).cpu(), density=True,bins=35)

xp = torch.linspace(.9*x.min(),1.1*x.max(),101).to(device)
lp = model.log_prob(xp.reshape(-1,1),torch.ones((101,5),device=device)*gt)
p = torch.exp(lp-lp.max())
p *= 1/(p.sum()*(xp[1]-xp[0]))
ax.plot(xp.cpu(),p.detach().cpu(),label='gt')

xp = torch.linspace(x.min(),x.max(),101).to(device)
lp = model.log_prob(xp.reshape(-1,1),torch.ones((101,5),device=device)*map_param)
p = torch.exp(lp-lp.max())
p *= 1/(p.sum()*(xp[1]-xp[0]))
ax.plot(xp.cpu(),p.detach().cpu(),label='map')
ax.legend()

ax.set_title('{} datapoints -- seed {}'.format(N,seed_data))

plt.savefig('FCYeast_mcmc/fig_comparison/mcmc_compare_{}_{}data_{}dp.png'.format(seed_mcmc,seed_data,N),dpi=600)

# %%




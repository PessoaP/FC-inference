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
N = 2**12

try:
    seed_data = int(sys.argv[1])
except:
    seed_data=42

seed_mcmc = 42
torch.manual_seed(seed_mcmc)

# %%
dv = np.loadtxt('FCYeast_synth/gt_map.csv')
gt = torch.tensor( dv[ dv[:,0]==seed_data ][0,1:] , device=device,dtype=torch.float)

x = np.loadtxt('FCYeast_synth/synth_{}.csv'.format(seed_data))[:N]
x = torch.tensor(x,device=device,dtype=torch.float).reshape(-1,1)
gt

# %%
FCYeast_simulator.adjust_device(device)
target = FCYeast_simulator.target()

model = architecture.make_model(device=device)
model_file = 'network.pt'

# %%
for param in model.parameters():
    param.requires_grad = False

# %%
try:
    model.load_state_dict(torch.load(model_file))
    print('loading pretrained network')
except:
    print('starting from scratch')

# %%
vectorize_params = torch.ones(1024,4).to(device)

def log_likelihood(data,params,model):
    global vectorize_params
    if data.size !=  vectorize_params.size(0):
        vectorize_params = torch.ones((data.size(0),4),device=device)

    return model.log_prob(data,params*vectorize_params)

def log_post(data,params,model,lprior):
    return log_likelihood(data,params,model).sum() + lprior(params)


# %%
lp_gt = log_post(x,gt,model,target.log_prior)
lp_gt

# %%
#first 100 from prior
params_100 = target.sample(N=100)[:,1:]
best_param = params_100[0]
lp_max = log_post(x,best_param,model,target.log_prior)

for par in params_100[1:]:
    lp_par = log_post(x,par,model,target.log_prior)
    if lp_par>lp_max:
        best_param = par
        lp_max=lp_par

for i in range(3,8):

    params_100 = (target.sample(N=100)[:,1:] - best_param)/i +best_param

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
S = torch.eye(4)*1e-4
mvn = torch.distributions.MultivariateNormal(torch.zeros(4,device=device),S.to(device))

def change_S(newS):
    global S
    global mvn

    S = newS
    mvn = torch.distributions.MultivariateNormal(torch.zeros(4,device=device),S.to(device))

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
        change_S( (torch.stack(sampled_params[-200:]).T.cov() + torch.eye(4)*1e-10) * ((2.4**2)/4)) 
    loopruns+=1

    print(loopruns,acc_rate,lp)
    



# # %%
# plt.plot(sampled_logpost)
# plt.axhline(lp_gt.item(),color='y')

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
np.savetxt('FCYeast_MCMC/results_{}data_{}dp.csv'.format(seed_data,N),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))


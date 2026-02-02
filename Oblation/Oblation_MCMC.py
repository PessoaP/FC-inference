# %%
# %%
import torch
import numpy as np
import normflows as nf
from tqdm import tqdm
import pandas as pd

# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)
torch.no_grad()

import sys
import os
c_directory = os.getcwd()
sys.path.append(os.path.join(c_directory, 'FCYeast2'))

from matplotlib import pyplot as plt
import FCYeast_extrinsic_simulator
import architecture

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

seed = int(sys.argv[1])
stress = sys.argv[2]
hereditary = sys.argv[3].lower() == "true"

torch.manual_seed(seed)
np.random.seed(seed)


# %%
g3 = np.loadtxt('../Real/FCYeast2_MCMC/mcmc_real_results_116224datapoints.csv')[-100000:,:]
map_index = np.argmax(g3[:,-1])
best_param = torch.tensor(g3[map_index]).float().to(device)
best_param


gt_high, gt_low = FCYeast_extrinsic_simulator.transform_to_arbitrary(best_param)
gt_high, gt_low = (gt_high).float(), (gt_low).float()

gt_sets = {"high": gt_high, "low": gt_low}

# %%
estimates = torch.load('../Real/ABC_estimates.pt',weights_only=False)
means_ABC = torch.tensor(estimates['training_means']).to(device)
sigmas_ABC = torch.tensor(estimates['training_sigmas']).to(device)

means_prior = FCYeast_extrinsic_simulator.transform_to_arbitrary(means_ABC).float()
prior_high = torch.distributions.MultivariateNormal((means_prior[0]).clone().detach().to(device), 
                                                    torch.diag((sigmas_ABC[:4])**2).clone().detach().to(device))
prior_low = torch.distributions.MultivariateNormal((means_prior[1]).clone().detach().to(device),    
                                                   torch.diag((sigmas_ABC[[0,4,5,6]])**2).clone().detach().to(device))

prior_high

# %%
dils_str = ['12','23']
dils = [.12,.23]
models=[]

for dil in dils_str:
    model = architecture.make_model()
    model_file = '../Real/dilution{}/network.pt'.format(dil)
    model.load_state_dict(torch.load(model_file))
    
    for param in model.parameters():
        param.requires_grad = False
    models.append(model)

model_high_stress, model_low_stress = models

# %%
if hereditary:
    dv_high = pd.read_csv(f'oblation_simulations/sim_high_seed{seed}.csv')
    dv_low  = pd.read_csv(f'oblation_simulations/sim_low_seed{seed}.csv')
else:
    dv_high = pd.read_csv(f'oblation_simulations/non_hereditary_sim_high_seed{seed}.csv')
    dv_low  = pd.read_csv(f'oblation_simulations/non_hereditary_sim_low_seed{seed}.csv')

data_high_stress = torch.tensor(dv_high['lI'].dropna().values).float().reshape(-1,1).to(device)
data_low_stress = torch.tensor(dv_low['lI'].dropna().values).float().reshape(-1,1).to(device)


# %%
vectorize_params = torch.ones(1024,4).to(device)

def log_likelihood(data,params,model):
    global vectorize_params
    if data.size !=  vectorize_params.size(0):
        vectorize_params = torch.ones((data.size(0),4),device=device)
    return model.log_prob(data,params*vectorize_params)

def log_post(data,params,model,prior):
    return log_likelihood(data,params,model).sum() + prior.log_prob(params)


# %%
outdir = "oblation_mcmc_results"
os.makedirs(outdir, exist_ok=True)


# %%

if stress=='low':
    model,x,prior,gt = model_low_stress,data_low_stress,prior_low,gt_low  
elif stress=='high':
    model,x,prior,gt = model_high_stress,data_high_stress,prior_high,gt_high



# %%
params_1k = prior.sample((1000,))
best_param = prior.loc
lp_max = log_post(x,best_param,model,prior)

for i in range(1,11):
    print(i)
    for par in ( (1/i)*(params_1k-prior.loc) + best_param ):
        lp_par = log_post(x,par,model,prior)
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
S = (3.14)*torch.eye(4)*1e-4
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
        lp_prop = log_post(x,param_prop,model,prior)

        if torch.log(torch.rand(1))< (lp_prop-lp).item():
            param = param_prop
            lp = lp_prop

        sampled_params.append(param.cpu())
        sampled_logpost.append(lp.cpu().item())

    #print(param,lp)

    acc_rate = np.mean([(sampled_params[i] - sampled_params[i-1]).sum().item()!=0 for i in range(-1,-101,-1)])

    if acc_rate>.2 and acc_rate<.5:
        count_of_safe += 1
    else:
        count_of_safe = 0

    if loopruns%3==2:
        change_S( (torch.stack(sampled_params[-200:]).T.cov() + torch.eye(4)*1e-8) * (2.4**2/(7)) )
    loopruns+=1

    print(loopruns,acc_rate, ' ', sampled_logpost[-1])#, 'priordist', (((best_param - target.prior.loc)/ target.prior.covariance_matrix.diag()).max()))
    

# %%
burnin = len(sampled_logpost)
for i in tqdm(range(100000)):
    param_prop = proposal(param)
    lp_prop = log_post(x,param_prop,model,prior)

    if torch.log(torch.rand(1))< (lp_prop-lp).item():
        param = param_prop
        lp = lp_prop

    sampled_params.append(param.cpu().numpy())
    sampled_logpost.append(lp.cpu().item())

    if i%100 == 99:
        print(i,param,lp, prior.log_prob(param))
        np.savetxt('{}/mcmc_results_{}_{}_{}.csv'.format(outdir,stress,seed,str(hereditary)),
           np.hstack((np.stack(sampled_params), np.array(sampled_logpost).reshape(-1,1))))
        


# %%




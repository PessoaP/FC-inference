# %%
import numpy as np
import torch

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import pandas as pd
import os
import sys

c_directory = os.getcwd()
sys.path.append(os.path.dirname(c_directory))

import architecture
import FCYeast_extrinsic_simulator 
import eZsamplers

device=FCYeast_extrinsic_simulator.device
l10=np.log(10)
FCYeast_extrinsic_simulator.adjust_device(device)



# %%
g3 = np.loadtxt('../Real/FCYeast2_MCMC/mcmc_real_results_116224datapoints.csv')[-100000:,:]
map_index = np.argmax(g3[:,-1])
best_param = torch.tensor(g3[map_index]).float().to(device)
best_param


# %%
outdir = "oblation_simulations"
os.makedirs(outdir, exist_ok=True)


#param_high, param_low = param[:4], param[[0, 4, 5, 6]]
param_high, param_low = FCYeast_extrinsic_simulator.transform_to_arbitrary(best_param)
param_high, param_low = torch.exp(param_high).float(), torch.exp(param_low).float()
#print(param_high, param_low)
param_sets = {"high": param_high, "low": param_low}
print(torch.exp(FCYeast_extrinsic_simulator.transform_to_hours(best_param)))

# %%
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    for pname, p in param_sets.items():
        # Run simulator
        t, Pr, s, betas, Zs = FCYeast_extrinsic_simulator.simulator(
            p[0], p[1:3], p[3], T=100, N=40000
        )
        
        # Convert Pr → intensity → total log-intensity
        I_prot = FCYeast_extrinsic_simulator.prot2intensity(Pr)
        autofluo = FCYeast_extrinsic_simulator.autofluo.sample(Pr.numel())[0].reshape(Pr.shape)#.clamp(min=1e-9)
        lI = torch.logaddexp(autofluo, torch.log(I_prot))



        df = pd.DataFrame({
            "Pr":     Pr.flatten().cpu().numpy(),
            "s":      s.flatten().cpu().numpy(),
            "betas":  betas.flatten().cpu().numpy(),
            "Zs":     Zs.flatten().cpu().numpy(),
            "lI":     lI.flatten().cpu().numpy(),
        })

        # Save
        fname = f"{outdir}/sim_{pname}_seed{seed}.csv"
        df.to_csv(fname, index=False)
        print(f"Saved: {fname}")

# %%




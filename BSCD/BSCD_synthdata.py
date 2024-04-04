import torch
import numpy as np
import BSCD_simulator

#from matplotlib import pyplot as plt
#from eZplot import BSCD_plot
#from tqdm import tqdm

import sys

N = 2**13
seed = int(sys.argv[1])
torch.manual_seed(seed)


enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BSCD_simulator.adjust_device(device)

target = BSCD_simulator.target()

if seed == 0:
    gt = target.params_dist.loc

else:
    gt = target.sample(n=1)[0][1:]

    dv = np.loadtxt('BSCD_synth/gt_map.csv')
    np.savetxt('BSCD_synth/gt_map.csv', np.vstack((dv,np.hstack((np.array((seed)),gt.cpu().numpy())))) )


lbeta,llam,lsig = gt[:2],gt[2:4],gt[-1],
x = target.sample(lbeta,llam,lsig,n=N)

#x = x[:,0].reshape(-1,1)
np.savetxt('BSCD_synth/synth_{}.csv'.format(seed),x[:,0].cpu().numpy())


import torch
import numpy as np
import FCYeast_simulator

import sys

N = 2**16
seed = int(sys.argv[1])
torch.manual_seed(seed)

enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

FCYeast_simulator.adjust_device(device)
target = FCYeast_simulator.target()

if seed == 0:
    gt = target.params_dist.loc
    np.savetxt('FCYeast_synth/gt_map.csv', np.hstack((np.array((seed)),gt.cpu().numpy())) )


else:
    gt = target.sample(n=1)[0][1:]

    dv = np.loadtxt('FCYeast_synth/gt_map.csv')
    np.savetxt('FCYeast_synth/gt_map.csv', np.vstack((dv,np.hstack((np.array((seed)),gt.cpu().numpy())))) )

lbeta,llam,lsig,lxi = gt[:1],gt[1:3],gt[3:4],gt[4:]
x = target.sample(lbeta,llam,lsig,lxi,n=N,return_lparams=False)

np.savetxt('FCYeast_synth/synth_{}.csv'.format(seed),x.cpu().numpy())


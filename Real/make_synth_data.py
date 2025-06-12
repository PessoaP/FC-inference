import torch 
import numpy as np
import FCYeast2_simulator


import sys

N = 2**16

seed = int(sys.argv[1])
torch.manual_seed(seed)


enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FCYeast2_simulator.adjust_device(device)

estimates = torch.load('ABC_estimates.pt')
means = estimates['training_means']
sigmas = estimates['training_sigmas']
target = FCYeast2_simulator.target(means,sigmas)


if seed == 0:
    gt = target.params_dist.loc
    np.savetxt('FCYeast2_synth/gt_map.csv', np.hstack((np.array((seed)),gt.cpu().numpy())) )


else:
    gt = torch.round(target.params_dist.sample((1,))[0],decimals=4)
    dv = np.loadtxt('FCYeast2_synth/gt_map.csv')
    np.savetxt('FCYeast2_synth/gt_map.csv', np.vstack((dv,np.hstack((np.array((seed)),gt.cpu().numpy())))) )


gt_sep = FCYeast2_simulator.transform_to_arbitrary(gt)

# lbetas,llams,lsigs= gt_sep[:,:1],gt_sep[:,1:3],gt_sep[:,3:4]
# lI = target.sample(lbetas,llams,lsigs,N=N)
lI = target.sample(gt_sep,N=N)

np.savetxt('FCYeast2_synth/synth_{}.csv'.format(seed),lI.cpu().numpy())




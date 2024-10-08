import torch 
import numpy as np
import FC3_simulator


import sys

N = 2**16

seed = int(sys.argv[1])
torch.manual_seed(seed)


enable_cuda = True
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

FC3_simulator.adjust_device(device)

target = FC3_simulator.target()


if seed == 0:
    gt = target.params_dist.loc
    np.savetxt('FC3_synth/gt_map.csv', np.hstack((np.array((seed)),gt.cpu().numpy())) )


else:
    gt = target.params_dist.sample((1,))[0]
    dv = np.loadtxt('FC3_synth/gt_map.csv')
    np.savetxt('FC3_synth/gt_map.csv', np.vstack((dv,np.hstack((np.array((seed)),gt.cpu().numpy())))) )


gt_sep = FC3_simulator.transform_to_arbitrary(gt)

lbetas,llams,lsigs= gt_sep[:,:1],gt_sep[:,1:3],gt_sep[:,3:4]
x = target.sample(lbetas,llams,lsigs,n=N)

np.savetxt('FC3_synth/synth_{}.csv'.format(seed),x.cpu().numpy())




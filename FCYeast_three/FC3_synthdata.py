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
    np.savetxt('FCYeast3_synth/gt_map.csv', np.hstack((np.array((seed)),gt.cpu().numpy())) )


else:
    gt = target.params_dist.sample((1,))[0]
    dv = np.loadtxt('FCYeast3_synth/gt_map.csv')
    np.savetxt('FCYeast3_synth/gt_map.csv', np.vstack((dv,np.hstack((np.array((seed)),gt.cpu().numpy())))) )

#print(gt)

gt_sep = FC3_simulator.transform_to_arbitrary(torch.stack([gt]))

lbetas,llams,lsigs,lxis = gt_sep[:,:,:2],gt_sep[:,:,2:4],gt_sep[:,:,4],gt_sep[:,:,5]
x = target.sample(lbetas,llams,lsigs,lxis,n=N)

np.savetxt('FCYeast3_synth/synth_{}.csv'.format(seed),x.cpu().numpy())




import numpy as np
from utils.motion_process import extract_root_trajectory_263

path = 'outputs/20260325_124453_ldf/HumanML3D/feature/000021.npy'
arr = np.load(path)
traj = extract_root_trajectory_263(arr)  # (T,3) root xyz


x = traj[:,0]
z = traj[:,2]
print('x min/max/mean/std:', x.min(), x.max(), x.mean(), x.std())
print('z min/max/mean/std:', z.min(), z.max(), z.mean(), z.std())

d = ((np.diff(x)**2 + np.diff(z)**2) ** 0.5)
print('per-frame ground disp mean/std:', d.mean(), d.std(), 'max:', d.max())
print('total ground path length:', d.sum())

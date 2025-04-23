import sys

import numpy as np

from get_collocation_points import *
print(sys.path)
import torch
np.random.seed(12)
torch.manual_seed(12)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_dtype(torch.float32)
from scipy.interpolate import griddata
from utils import *
from PINN_PF_class import PINN_PotFlow_HC_pBC

name_save = f'irreg_HC_PFT_example'

# learning rates
lr_LBFGS = 1
lr_Adam = 0.0005

# optimizer hyperparameter
epochsAdam = 5000
epochsLBFGS = 40000 - epochsAdam
hist_size = 20

# nodes per layer  (2 or 3 inputs, 21 for Fourier-feature-embedding)
layers_eta = [2 * 21, 200, 200, 200, 200, 1]
layers_phi = [3 * 21, 200, 200, 200, 200, 1]

# hyperparameter for ReLoBraLo loss-balancing
rho = 0.98
alpha = 0.95
tau = 20

# scaling for Fourier-feature embedding
scaling_freq = 0.4
scales = np.linspace(-scaling_freq, scaling_freq, 10)

# initial loss-weights
w_data = 1
w_pde = 1
w_bc_kin = 1
w_bc_dyn = 1
w_bc_bot = 1
w_pb = 1

# number of collocation points (try smaller numbers if training not possible)
N_bc_sur = 5000  # random collocation points for surface BCs
N_bc_bot = 1000  # random collocation points for bottom BC
N_pde = 30000    # random collocation points inside the domain for Laplace Eq.
N_pb = 1000      # points for periodic boundary conditions

# Load the ground truth wave data (for comparison of the PINN's reconstruction post-training)
LWT_data = np.load('LWT_surface_data.npz')
Eta_true = LWT_data['eta']
Phi_true = LWT_data['phi']
x = LWT_data['x']
t = LWT_data['t']
z = LWT_data['z']
d = np.abs(np.min(z))
X, T, Z = np.meshgrid(x, t, z)

# indices of buoy positions
meas1 = 0
meas2 = int(len(x) / 2)
meas3 = len(x) - 1

# extract sparse measurement data from elevation and corresponding coordinates in x-t-plane (only data provided for PINN training)
id_data = np.random.choice(t.shape[0], len(x) - 1, replace=False)
xt_data = np.vstack((np.hstack((X[:, meas1:meas1 + 1, -1], T[:, meas1:meas1 + 1, -1]))[id_data, :],
                     np.hstack((X[:, meas2:meas2 + 1, -1], T[:, meas2:meas2 + 1, -1]))[id_data, :],
                     np.hstack((X[:, meas3:meas3 + 1, -1], T[:, meas3:meas3 + 1, -1]))[id_data, :]))
eta_data = np.vstack((Eta_true[:, meas1:meas1 + 1][id_data, :],
                      Eta_true[:, meas2:meas2 + 1][id_data, :],
                      Eta_true[:, meas3:meas3 + 1][id_data, :]))

# get the collocation points
xtz_pde, xtz_pde_up, xt_bc_sur, xtz_bc_bot, xtz_lb, xtz_ub, lb, ub = \
    get_collocation_points(x=x, t=t, z=z,
                           eta_meas_max=np.max(np.abs(eta_data)),
                           N_pde=N_pde,
                           N_bc_sur=N_bc_sur,
                           N_bc_bot=N_bc_bot,
                           N_pb=N_pb)

# equally spaced points for evaluation
X_star = np.hstack((X[:, :, -1].flatten()[:, None], T[:, :, -1].flatten()[:, None], Z[:, :, -1].flatten()[:, None]))

# dictionaries for PFT-PINN model
dict_loss_weights = {'w_data': w_data, 'w_pde': w_pde, 'w_bc_kin': w_bc_kin,
                     'w_bc_dyn': w_bc_dyn, 'w_bc_bot': w_bc_bot, 'w_pb': w_pb}

dict_collpoints = {'xt_data': xt_data, 'eta_data': eta_data,
                   'xtz_pde': xtz_pde, 'xtz_pde_up': xtz_pde_up,
                   'xt_bc_sur': xt_bc_sur, 'xtz_bc_bot': xtz_bc_bot,
                   'xtz_ub': xtz_ub, 'xtz_lb': xtz_lb,
                   'lb': lb, 'ub': ub, 'X_star': X_star}

# initialize PFT-PINN model
model = PINN_PotFlow_HC_pBC(dic=dict_collpoints, l_w_dic=dict_loss_weights,
                            neurons_lay_eta=layers_eta, neurons_lay_phi=layers_phi,
                            hs_LBFGS=hist_size, epAdam=epochsAdam, epLBFGS=epochsLBFGS,
                            lr_LBFGS=lr_LBFGS, lr_Adam=lr_Adam, name_for_save=name_save,
                            alpha=alpha, tau=tau, rho=rho, scales=scales)

# train the model
model.train()

# predict reconstruction of elevation and potential at z=0
Eta_pred_, Phi_pred_ = model.inference_eta_phi(x=X_star[:, 0:1], t=X_star[:, 1:2], z=X_star[:, 2:3])

# interpolate to shape of the meshgrid
Eta_pred = griddata(X_star[:, 0:2], Eta_pred_.flatten(), (X[:, :, -1], T[:, :, -1]), method='cubic')
Phi_pred = griddata(X_star[:, 0:2], Phi_pred_.flatten(), (X[:, :, -1], T[:, :, -1]), method='cubic')

# plot the loss-curve
plotting_losscurve_PB(path_loss=model.LOSS_PATH, path_save=model.FIGURE_PATH, format_save='png',
                      xmax=epochsAdam + epochsLBFGS, figsize=(7, 2.6),
                      ymax=10, ymin=0.0000001)

# plot elevation and potential
plotting_PINN_elevation(x=x, t=t, eta_true=Eta_true, eta_pred=Eta_pred,
                        xt_train=np.vstack([xt_data]),
                        x_is=np.linspace(0, len(x) - 1, 5).astype(int), epoch=epochsLBFGS + epochsAdam,
                        path_save=model.FIGURE_PATH, format_save='pdf')

Phi_pred = shift_phi_to_match_mean(Phi_pred, 0)
plotting_PINN_potential(x=x, t=t, phi_pred=Phi_pred, phi_true=Phi_true,
                        x_is=np.linspace(0, len(x) - 1, 5).astype(int), epoch=epochsLBFGS + epochsAdam,
                        path_save=model.FIGURE_PATH, format_save='pdf')

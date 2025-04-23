import numpy as np
from pyDOE import lhs
np.random.seed(12)


def get_collocation_points(x, t, z, eta_meas_max, N_pde, N_bc_sur, N_bc_bot, N_pb):

    d = np.max(np.abs(z))   # water depth

    # domain bounds total
    lb = np.array([np.min(x), np.min(t), np.min(z)])                       # lower boundary: [X_min, T_min, Z_min]
    ub = np.array([np.max(x), np.max(t), np.max(z) + 1.1 * eta_meas_max])  # upper boundary: [X_max, T_max, Z_max + 1.1 eta_max]

    # fixed collocation points inside the domain for Laplace equation
    lb_ = np.array([np.min(x), np.min(t), np.min(z)])
    ub_ = np.array([np.max(x), np.max(t), 0 - 1.1 * eta_meas_max])
    xtz_pde = lb_ + (ub_ - lb_) * lhs(3, int(N_pde * 0.8))

    # adaptive collocation points for Laplace equation near the wave-air interface (changed during training according to instantaneous free surface)
    lb_ = np.array([np.min(x), np.min(t), 0 - 1.1 * eta_meas_max])
    ub_ = np.array([np.max(x), np.max(t), 0 + 1.1 * eta_meas_max])
    xtz_pde_up = lb_ + (ub_ - lb_) * lhs(3, int(N_pde * 0.2))

    # collocation points for surface BCs (no z-component, as this is z=eta during training)
    xt_bc_sur = lb[0:2] + (ub[0:2] - lb[0:2]) * lhs(2, N_bc_sur)

    # collocation points for bottom BC
    xtz_bc_bot = lb[0:2] + (ub[0:2] - lb[0:2]) * lhs(2, N_bc_bot)
    xtz_bc_bot = np.hstack((xtz_bc_bot, -d * np.ones(shape=(N_bc_bot, 1))))

    # collocation points for periodic BCs in time
    xtz_phi = lb + (ub - lb) * lhs(3, N_pb)
    xtz_lb = np.hstack((xtz_phi[:, 0:1], np.ones(shape=(N_pb, 1)) * lb[1], xtz_phi[:, 2:]))
    xtz_ub = np.hstack((xtz_phi[:, 0:1], np.ones(shape=(N_pb, 1)) * ub[1], xtz_phi[:, 2:]))

    return xtz_pde, xtz_pde_up, xt_bc_sur, xtz_bc_bot, xtz_lb, xtz_ub, lb, ub

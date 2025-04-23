import sys

print(sys.path)
import torch
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
from libs.helper_functions import *
import time
from csv import writer
from os.path import exists
from itertools import chain
import os


def multiscale(inp, scales):
    """
    Generates multiscale frequency projections for Fourier feature embedding
    :param inp: input tensor
    :param scales: list of frequency scaling factors
    """
    ms = torch.hstack([inp.reshape(-1, 1) * 2 * torch.pi * i for i in scales])
    return ms


def encode_scalar_column(inp_norm, scales):
    """
    creates Fourier feature embedding for a single input dimension by multiscale frequency projection (sin/cos components)
    and concatenation with original normalized input
    :param inp_norm: Pre-normalized input tensor
    :param scales: list of frequency scaling factors
    """
    ms = multiscale(inp_norm, scales)
    back = torch.hstack([torch.sin(ms), torch.cos(ms), inp_norm[:, None]])
    return back


def init_weight_bias(m):
    """initializes weights of a NN layer with Xavier distribution """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def cdn(var: torch.Tensor):
    """function converts tensors on GPU back to numpy arrays on CPU"""
    return var.cpu().detach().numpy()


def tfn(var: np.ndarray):
    """generates torch tensor from numpy in float32 on GPU device"""
    return torch.from_numpy(var).float().to(device)


def tfl(list):
    """generates torch tensor from list on GPU device"""
    return torch.Tensor(list).to(device)


def tff(var: np.ndarray):
    """generates torch tensor from float in float32 on GPU device"""
    return torch.Tensor([var]).to(device)


def write_csv_line(path: str, line):
    """ writes a new line to a csv file at path"""
    with open(path, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(line)


class Net_Sequential_F(nn.Module):
    def __init__(self, lb, ub, nodes, scales):
        """
        Multilayer Perceptron with learnable Fourier feature embedding
        :param lb: lower bounds for input normalization
        :param ub: upper bounds for input normalization
        :param nodes: list of number of units in each layer (including input and output)
        :param scales: initial frequency scaling factors for Fourier feature embedding
        """
        super(Net_Sequential_F, self).__init__()
        self.lb = lb
        self.ub = ub
        self.nodes = nodes
        self.layers = nn.ModuleList()

        # initialize learnable frequency scales
        sequence = np.array(scales)[:, None].repeat(len(self.lb), 1)
        self.scales = nn.Parameter(tfn(sequence), requires_grad=True)

        # build NN layers
        for i in range(len(self.nodes) - 1):
            self.layers.append(nn.Linear(int(self.nodes[i]), int(self.nodes[i + 1])))
        self.tanh = nn.Tanh()

    def forward(self, *inp):
        """
        Forward pass with Fourier feature transformation
        :param inp: input tensor is tuple, either (x, t) or (x, t, z)
        :return: final network output
        """
        inputs_raw = torch.cat(inp, axis=1)

        # normalize to [-1, 1]
        inputs_norm = torch.div(2.0 * (inputs_raw - self.lb), self.ub - self.lb) - 1.0

        # Apply Fourier feature embedding per input dimension
        inputs = torch.cat([encode_scalar_column(inp_norm=inputs_norm[:, i], scales=self.scales[:, i]) for i in range(inputs_raw.shape[1])], dim=1)

        # hidden layers with tanh activation
        for i in range(len(self.nodes) - 2):
            lay = self.layers[i]
            inputs = self.tanh(lay(inputs))

        # final linear layer
        lay = self.layers[i + 1]
        output = lay(inputs)

        return output


class Net_Sequential(nn.Module):
    def __init__(self, lb, ub, nodes):
        """
        normal Multilayer Perceptron
        :param lb: lower bounds for input normalization
        :param ub: upper bounds for input normalization
        :param nodes: list of number of units in each layer (including input and output)
        """
        super(Net_Sequential, self).__init__()
        self.lb = lb
        self.ub = ub
        self.nodes = nodes
        self.layers = nn.ModuleList()

        # build NN layers
        for i in range(len(self.nodes) - 1):
            self.layers.append(nn.Linear(int(self.nodes[i]), int(self.nodes[i + 1])))
        self.tanh = nn.Tanh()

    def forward(self, *inp):
        """
        Forward pass
        :param inp: input tensor is tuple, either (x, t) or (x, t, z)
        :return: final network output
        """
        inputs = torch.cat(inp, axis=1)

        # normalize to [-1, 1]
        inputs = torch.div(2.0 * (inputs - self.lb), self.ub - self.lb) - 1.0

        # hidden layers with tanh activation
        for i in range(len(self.nodes) - 2):
            lay = self.layers[i]
            inputs = self.tanh(lay(inputs))

        # final linear layer
        lay = self.layers[i + 1]
        output = lay(inputs)

        return output


class PINN_PotFlow_HC_pBC:
    def __init__(self, dic, l_w_dic, neurons_lay_eta, neurons_lay_phi, hs_LBFGS, lr_LBFGS, lr_Adam,
                 epAdam, epLBFGS, name_for_save, alpha, tau, rho, scales):
        """
        class to implement and train a Physics-informed Neural Network for the Potential Flow Theory of ocean gravity waves
        :param dic: dictionary containing the
        :param l_w_dic: dictionary containing the chosen loss initial loss weights for each loss components
        :param neurons_lay_eta: list of neurons per layer for elevation network eta
        :param neurons_lay_phi: list of neurons per layer for potential network phi
        :param hs_LBFGS: history size for the LBFGS optimizer
        :param lr_LBFGS: learning rate for the LBFGS optimizer
        :param lr_Adam: learning rate for the Adam optimizer
        :param epAdam: epochs of Adam optimizer
        :param epLBFGS: maximum of iterations/epochs for the LBFGS optimizer is equal to max_it! So no training-for-loop needed later.
        :param name_for_save: string with the name for saving the model parameters and figures
        :param alpha: parameter of ReLoBRaLo balancing historical weight information with new updates
        :param tau: sensitivity of weight assignment in ReLoBRaLo
        :param rho: bernoulli random variable for ReLoBRaLo to introduce random look-backs
        :param scales: list of frequency scaling factors for Fourier feature embedding
        """

        # initialize basics
        self.loss, self.loss_scaled, self.loss_MSE_bc_kin, self.loss_MSE_bc_dyn, self.loss_MSE_bc_bot, self.loss_MSE_data, self.loss_MSE_pde, self.loss_MSE_pb_phi, self.loss_MSE_pb_eta = [None] * 9
        self.name_save = name_for_save
        self.epochsAdam, self.epochsLBFGS = epAdam, epLBFGS
        self.epoch, self.start_time = 0, time.time()
        self.best_loss = torch.inf

        # create folders for saving
        self.MODEL_PATH, self.ERROR_PATH, self.FIGURE_PATH = f'models/{name_for_save}.pth', f"errors/{name_for_save}/", f"figures/{name_for_save}/"
        os.makedirs(self.ERROR_PATH, exist_ok=True)
        os.makedirs(self.FIGURE_PATH, exist_ok=True)
        self.LOSS_PATH = os.path.join(self.ERROR_PATH, f"loss.csv")

        # for ReLoBRaLo
        self.last_losses, self.init_losses, self.last_weights, self.losses_ep, self.weights_ep = [None] * 5
        self.alpha = tff(alpha)
        self.tau = tff(tau)
        self.rho = tff(rho)

        # initial values of adaptive loss weighting factors
        self.w_data = Variable(tff(l_w_dic['w_data']), requires_grad=False)
        self.w_pde = Variable(tff(l_w_dic['w_pde']), requires_grad=False)
        self.w_bc_dyn = Variable(tff(l_w_dic['w_bc_dyn']), requires_grad=False)
        self.w_bc_kin = Variable(tff(l_w_dic['w_bc_kin']), requires_grad=False)
        self.w_bc_bot = Variable(tff(l_w_dic['w_bc_bot']), requires_grad=False)
        self.w_pb_eta = Variable(tff(l_w_dic['w_pb']), requires_grad=False)
        self.w_pb_phi = Variable(tff(l_w_dic['w_pb']), requires_grad=False)

        # domain bounds
        self.lower_bound, self.upper_bound = tfn(dic['lb']), tfn(dic['ub'])

        # sparse sensor data
        self.x_data = Variable(tfn(dic['xt_data'][:, 0:1]), requires_grad=True)
        self.t_data = Variable(tfn(dic['xt_data'][:, 1:2]), requires_grad=True)
        self.eta_data_true = Variable(tfn(dic['eta_data']), requires_grad=False)

        # collocation points at boundary (x, t, z=0) for free surface boundary conditions
        self.x_bc_sur = Variable(tfn(dic['xt_bc_sur'][:, 0:1]), requires_grad=True)
        self.t_bc_sur = Variable(tfn(dic['xt_bc_sur'][:, 1:2]), requires_grad=True)

        # collocation points at boundary (x, t, z=-d) for bed boundary condition
        self.x_bc_bot = Variable(tfn(dic['xtz_bc_bot'][:, 0:1]), requires_grad=True)
        self.t_bc_bot = Variable(tfn(dic['xtz_bc_bot'][:, 1:2]), requires_grad=True)
        self.z_bc_bot = Variable(tfn(dic['xtz_bc_bot'][:, 2:3]), requires_grad=True)

        # fixed collocation points for Laplace equation inside entire (x, t, z)-domain with z-values far enough from the free surface
        self.x_pde = Variable(tfn(dic['xtz_pde'][:, 0:1]), requires_grad=True)
        self.t_pde = Variable(tfn(dic['xtz_pde'][:, 1:2]), requires_grad=True)
        self.z_pde = Variable(tfn(dic['xtz_pde'][:, 2:3]), requires_grad=True)

        # adaptive collocation points for Laplace equation inside entire (x, t, z)-domain with z-values being adaptive according to instantaneous surface elevation
        self.x_pde_up = Variable(tfn(dic['xtz_pde_up'][:, 0:1]), requires_grad=True)
        self.t_pde_up = Variable(tfn(dic['xtz_pde_up'][:, 1:2]), requires_grad=True)
        self.z_pde_up = Variable(tfn(dic['xtz_pde_up'][:, 2:3]), requires_grad=True)

        # collocation points at boundaries (x, t=t_min, z) or (x, t=t_max, z) for periodic boundary conditions
        self.x_lb = Variable(tfn(dic['xtz_lb'][:, 0:1]), requires_grad=True)
        self.t_lb = Variable(tfn(dic['xtz_lb'][:, 1:2]), requires_grad=True)
        self.z_lb = Variable(tfn(dic['xtz_lb'][:, 2:3]), requires_grad=True)
        self.x_ub = Variable(tfn(dic['xtz_ub'][:, 0:1]), requires_grad=True)
        self.t_ub = Variable(tfn(dic['xtz_ub'][:, 1:2]), requires_grad=True)
        self.z_ub = Variable(tfn(dic['xtz_ub'][:, 2:3]), requires_grad=True)

        # values for intermediate plotting
        self.X_star = dic['X_star']

        # Initialize NNs
        self.model_eta = Net_Sequential_F(lb=self.lower_bound[0:2], ub=self.upper_bound[0:2], nodes=neurons_lay_eta, scales=scales)
        self.model_phi = Net_Sequential_F(lb=self.lower_bound, ub=self.upper_bound, nodes=neurons_lay_phi, scales=scales)
        self.model_eta_HC = Net_Sequential(lb=self.lower_bound[0:2], ub=self.upper_bound[0:2], nodes=[2, 50, 50, 1])
        self.model_distance_fun = Net_Sequential(lb=self.lower_bound[0:2], ub=self.upper_bound[0:2], nodes=[2, 50, 50, 1])

        # Xavier weight initialization
        self.model_eta.apply(init_weight_bias)
        self.model_phi.apply(init_weight_bias)
        self.model_eta_HC.apply(init_weight_bias)
        self.model_distance_fun.apply(init_weight_bias)

        # send to GPU
        self.model_eta.float().to(device)
        self.model_phi.float().to(device)
        self.model_eta_HC.float().to(device)
        self.model_distance_fun.float().to(device)

        # initialize optimizers
        self.optimizerAdam = torch.optim.Adam(chain(self.model_eta.parameters(), self.model_phi.parameters()), lr=lr_Adam, amsgrad=True)
        self.optimizerLBFGS = torch.optim.LBFGS(chain(self.model_eta.parameters(), self.model_phi.parameters()),
                                                lr=lr_LBFGS,
                                                max_iter=self.epochsLBFGS,
                                                history_size=hs_LBFGS,
                                                line_search_fn=None,
                                                tolerance_grad=1e-200, tolerance_change=1e-400)
        self.optimizer_DF = torch.optim.Adam(self.model_distance_fun.parameters(), lr=0.0001)
        self.optimizer_HC = torch.optim.Adam(self.model_eta_HC.parameters(), lr=0.0001)

    def net_eta(self, x, t, cond):
        """
        computes surface elevation with derivatives and implements hard constraints approach
        :param x: spatial coordinates
        :param t: temporal coordinates
        :param cond: condition mode
        :return: surface elevation and its derivatives in x and t direction at point (x,t)
        """
        # base eta-network output
        eta_u = self.model_eta(x, t)

        # apply hard constraint modification
        eta = self.model_eta_HC(x, t) + self.model_distance_fun(x, t) * eta_u

        if cond == 'infer':
            # no derivatives needed for pure inference/forward pass
            eta_x = 0
            eta_t = 0
        elif cond == 'sur':
            # derivatives for surface BCs
            eta_x = torch.autograd.grad(eta.sum(), x, create_graph=True)[0]
            eta_t = torch.autograd.grad(eta.sum(), t, create_graph=True)[0]

        return eta, eta_x, eta_t

    def net_phi(self, x, t, z, cond):
        """
        computes velocity potential with derivatives for different conditions
        :param x: spatial coordinates
        :param t: temporal coordinates
        :param z: vertical coordinates
        :param cond: condition mode
        :return: velocity potential and its derivatives in x, z and t direction at point (x,t,z)
        """

        # base phi-network output
        phi = self.model_phi(x, t, z)

        if cond == 'infer':
            # no derivatives needed for pure inference/forward pass
            phi_x = 0
            phi_t = 0
            phi_z = 0
            phi_xx = 0
            phi_zz = 0

        elif cond == 'pde':
            # derivatives for Laplace equation
            phi_x = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
            phi_t = 0
            phi_z = torch.autograd.grad(phi.sum(), z, create_graph=True)[0]
            phi_xx = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0]
            phi_zz = torch.autograd.grad(phi_z.sum(), z, create_graph=True)[0]

        elif cond == 'sur':
            # derivatives for surface BCs
            phi_x = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
            phi_t = torch.autograd.grad(phi.sum(), t, create_graph=True)[0]
            phi_z = torch.autograd.grad(phi.sum(), z, create_graph=True)[0]
            phi_xx = 0
            phi_zz = 0

        elif cond == 'bot':
            # derivatives for bottom BC
            phi_x = 0
            phi_t = 0
            phi_z = torch.autograd.grad(phi.sum(), z, create_graph=True)[0]
            phi_xx = 0
            phi_zz = 0

        return phi, phi_x, phi_t, phi_z, phi_xx, phi_zz

    def loss_PINN(self):
        """
        calculates the physics-informed loss function for Potential Flow Theory
        and additionally calculates the adaptive loss-component weighting factor with ReLoBRaLO
        :return: total loss function as a sum of the weighted components
        """
        # eta-prediction for collocation points at buoy positions
        eta_data_pred, _, _ = self.net_eta(x=self.x_data, t=self.t_data, cond='infer')

        # eta-prediction and derivatives for collocation points on surface
        eta_bc_sur, eta_x_bc_sur, eta_t_bc_sur = self.net_eta(x=self.x_bc_sur, t=self.t_bc_sur, cond='sur')

        # phi-derivatives for collocation points on surface, note that the z-input is the free surface elevation
        _, phi_x_bc_sur, phi_t_bc_sur, phi_z_bc_sur, _, _ = self.net_phi(x=self.x_bc_sur, t=self.t_bc_sur, z=eta_bc_sur, cond='sur')

        # phi-derivative for collocation points on seabed
        _, _, _, phi_z_bc_bot, _, _ = self.net_phi(x=self.x_bc_bot, t=self.t_bc_bot, z=self.z_bc_bot, cond='bot')

        # handling of collocation points directly beneath the free surface: if z>eta than the z value is replaced by eta
        eta_pde_up, _, _ = self.net_eta(x=self.x_pde_up, t=self.t_pde_up, cond='infer')
        z_pde_up = torch.where(self.z_pde_up > eta_pde_up, input=eta_pde_up, other=self.z_pde_up)

        # phi-derivatives for collocation points in the entire domain
        _, _, _, _, phi_xx_pde, phi_zz_pde = self.net_phi(x=torch.cat((self.x_pde, self.x_pde_up)),
                                                          t=torch.cat((self.t_pde, self.t_pde_up)),
                                                          z=torch.cat((self.z_pde, z_pde_up)), cond='pde')

        # eta- and phi-prediction at domain boundaries
        eta_lb, _, _ = self.net_eta(x=self.x_lb, t=self.t_lb, cond='infer')
        eta_ub, _, _ = self.net_eta(x=self.x_ub, t=self.t_ub, cond='infer')
        phi_lb, _, _, _, _, _ = self.net_phi(x=self.x_lb, t=self.t_lb, z=self.z_lb, cond='infer')
        phi_ub, _, _, _, _, _ = self.net_phi(x=self.x_ub, t=self.t_ub, z=self.z_ub, cond='infer')

        # individual loss components (data, BC_kin, BC_dyn, BC_bot, Laplace, periodic BCs eta, periodic BCs phi)
        self.loss_MSE_data = torch.mean(torch.square(self.eta_data_true - eta_data_pred))
        self.loss_MSE_bc_kin = torch.mean(torch.square(eta_t_bc_sur + eta_x_bc_sur * phi_x_bc_sur - phi_z_bc_sur))
        self.loss_MSE_bc_dyn = torch.mean(torch.square(phi_t_bc_sur + 9.81 * eta_bc_sur + 0.5 * (torch.square(phi_x_bc_sur) + torch.square(phi_z_bc_sur))))
        self.loss_MSE_bc_bot = torch.mean(torch.square(phi_z_bc_bot))
        self.loss_MSE_pde = torch.mean(torch.square(phi_xx_pde + phi_zz_pde))
        self.loss_MSE_pb_eta = torch.mean(torch.square(eta_lb - eta_ub))
        self.loss_MSE_pb_phi = torch.mean(torch.square(phi_lb - phi_ub))

        # ReLoBRaLo loss balancing scheme
        if self.epoch <= 10:
            # initialization phase
            self.init_losses = [self.loss_MSE_data, self.loss_MSE_bc_kin, self.loss_MSE_bc_dyn, self.loss_MSE_bc_bot, self.loss_MSE_pde,
                                self.loss_MSE_pb_eta, self.loss_MSE_pb_phi]
            self.last_losses = [self.loss_MSE_data, self.loss_MSE_bc_kin, self.loss_MSE_bc_dyn, self.loss_MSE_bc_bot, self.loss_MSE_pde,
                                self.loss_MSE_pb_eta, self.loss_MSE_pb_phi]
            self.last_weights = tfl([self.w_data, self.w_bc_kin, self.w_bc_dyn, self.w_bc_bot, self.w_pde, self.w_pb_eta, self.w_pb_phi])
        else:
            rho_ = torch.bernoulli(self.rho).to(device) # bernoulli random variable for this epoch either 0 or 1

            self.losses_ep = [self.loss_MSE_data, self.loss_MSE_bc_kin, self.loss_MSE_bc_dyn, self.loss_MSE_bc_bot, self.loss_MSE_pde,
                              self.loss_MSE_pb_eta, self.loss_MSE_pb_phi]

            # loss-scaling relative to last epoch
            input_hat = tfl([self.losses_ep[i] / (self.last_losses[i] * self.tau + 1e-9) for i in range(len(self.losses_ep))])
            weights_hat = len(self.losses_ep) * torch.nn.functional.softmax(input_hat - torch.max(input_hat), dim=0)

            # loss scaling relative to initial loss value
            input0_hat = tfl([self.losses_ep[i] / (self.init_losses[i] * self.tau + 1e-9) for i in range(len(self.losses_ep))])
            weights0_hat = len(self.losses_ep) * torch.nn.functional.softmax(input0_hat - torch.max(input0_hat), dim=0)

            # weight update rule
            self.weights_ep = self.alpha * (rho_ * self.last_weights + (1 - rho_) * weights0_hat) + (1 - self.alpha) * weights_hat

            # unpack weights for readability and monitoring
            self.w_data, self.w_bc_kin, self.w_bc_dyn, self.w_bc_bot, self.w_pde, self.w_pb_eta, self.w_pb_phi = self.weights_ep[0], self.weights_ep[1], self.weights_ep[2], \
                                                                                                                 self.weights_ep[3], self.weights_ep[4], self.weights_ep[5], \
                                                                                                                 self.weights_ep[6]
            # update for next epoch
            self.last_weights = self.weights_ep
            self.last_losses = self.losses_ep

        # final total loss weighted by loss weighting factors
        self.loss_scaled = self.w_data * self.loss_MSE_data + self.w_bc_kin * self.loss_MSE_bc_kin \
                           + self.w_bc_dyn * self.loss_MSE_bc_dyn + self.w_bc_bot * self.loss_MSE_bc_bot \
                           + self.w_pde * self.loss_MSE_pde + self.w_pb_eta * self.loss_MSE_pb_eta + self.w_pb_phi * self.loss_MSE_pb_phi

        # loss without scaling for monitoring
        self.loss = self.loss = self.loss_MSE_data + self.loss_MSE_bc_kin + self.loss_MSE_bc_dyn \
                    + self.loss_MSE_bc_bot + self.loss_MSE_pde + self.loss_MSE_pb_eta + self.loss_MSE_pb_phi

        return self.loss_scaled

    def pretrain_hard_constraint(self):
        """
        Pretrains sub-networks to enforce hard constraints for PINN
        1. Pretrains the η_HC network (model_eta_HC) to fit the buoy data exactly
        2. Pretrains the distance function network (model_distance_fun) to smoothly approximate the minimum
           distance from each (x, t) surface/boundary point to the nearest buoy data point.
        """

        # 1. train network M(x,t) that approximates/fits the buoy data
        for e in range(11000):
            self.optimizer_HC.zero_grad()
            eta_data_pred = self.model_eta_HC(self.x_data, self.t_data)
            loss_HC = torch.mean(torch.square(self.eta_data_true - eta_data_pred))
            loss_HC.backward(retain_graph=True)
            self.optimizer_HC.step()
            if e % 10 == 0:
                print(f'epoch: {e}, MSE_data for eta_HC pretrain: {loss_HC}')

        # 2. calculate distance function r(x,t) analytically (minimum distance between each surface point to each buoy data point)
        x_dist = torch.cat((self.x_bc_sur, self.x_data), dim=0)
        t_dist = torch.cat((self.t_bc_sur, self.t_data), dim=0)
        # compute pairwise Euclidean distances between each (x, t) and all data points
        min_distances = torch.min(torch.sqrt(torch.square(x_dist - torch.transpose(self.x_data, 0, 1)) +
                                             torch.square(t_dist - torch.transpose(self.t_data, 0, 1))), dim=1)
        min_distances = min_distances[0][:, None]
        min_distances = min_distances / (torch.max(torch.max(min_distances)))   # normalize

        # 3. train the distance function model R(x,t) to approximate the calculated distances r(x,t) smoothly
        for e in range(50000):
            self.optimizer_DF.zero_grad()
            distance_fun_pred = self.model_distance_fun(x_dist, t_dist)
            loss_DF = torch.mean(torch.square(distance_fun_pred - min_distances))
            loss_DF.backward(retain_graph=True)
            self.optimizer_DF.step()
            if e % 10 == 0:
                print(f'epoch: {e}, MSE_data for distance function pretrain: {loss_DF}')

    def train(self):
        """
        function to train (or load if trained before) the model defined by name_save:
        first pretrains the low-capacity networks for the hard constraints, then using
        Adam and LBFGS optimizer for specified number of epochs for the general PINN solution
        """
        if exists(self.MODEL_PATH):
            # load model if it was trained before
            self.load_model(path=self.MODEL_PATH)
        else:
            # start the training procedure
            def adam_closure():
                self.optimizerAdam.zero_grad()
                loss = self.loss_PINN()
                loss.backward(retain_graph=True)
                self.callback()
                return loss

            def lbfgs_closure():
                self.optimizerLBFGS.zero_grad()
                loss = self.loss_PINN()
                if torch.isnan(loss):
                    raise ValueError("NaN value encountered during LBFGS optimization")
                if loss.requires_grad:
                    loss.backward(retain_graph=True)
                self.callback()
                return loss

            # pretrain the hard constraints for the measurement locations
            self.pretrain_hard_constraint()

            # Adam optimizer (in loop)
            print(f'\n\n\nAdam Optimizer for {self.epochsAdam} epochs: \n\n\n')
            for epoch in range(self.epochsAdam):
                self.optimizerAdam.step(adam_closure)

            # LBFGS optimizer (not in loop as max number of epochs already specified in optimizer initialization)
            print(f'\n\n\nL-BFGS Optimizer for {self.epochsLBFGS} epochs: \n\n\n')
            self.optimizerLBFGS.step(lbfgs_closure)

            # load best model at the end of training
            self.load_model(path=self.MODEL_PATH)

    def callback(self):
        """
        prints the loss components of current epoch to console and saves them to loss csv-file
        prints the loss weighting factors (from ReLoBRaLo) and scales (from Fourier feature embedding) to console
        checks if a current epoch's loss is better than observed before and if so - saves model
        """
        elapsed_time = time.time() - self.start_time

        # print loss components for current epoch
        keys = ['epoch', 'loss', 'loss_scaled', 'MSE_data', 'MSE_pde', 'MSE_bc_kin', 'MSE_bc_dyn', 'MSE_bc_bot', 'MSE_pb_eta', 'MSE_pb_phi']
        vals = [self.epoch, cdn(self.loss), cdn(self.loss_scaled), cdn(self.loss_MSE_data), cdn(self.loss_MSE_pde),
                cdn(self.loss_MSE_bc_kin), cdn(self.loss_MSE_bc_dyn), cdn(self.loss_MSE_bc_bot), cdn(self.loss_MSE_pb_eta),
                cdn(self.loss_MSE_pb_phi)]
        print(f'time per epoch: {np.round(elapsed_time, 3)} s')
        print("".join(str(key) + ": " + str(value) + ", " for key, value in zip(keys, vals)))

        if self.epoch % 50 == 0:
            # print loss weighting factors and scales of the Fourier feature embedding for current epoch
            keys_w = ['w_data', 'w_pde', 'w_bc_kin', 'w_bc_dyn', 'w_bc_bot', 'w_pb_eta', 'w_pb_phi']
            vals_w = [cdn(self.w_data), cdn(self.w_pde), cdn(self.w_bc_kin), cdn(self.w_bc_dyn), cdn(self.w_bc_bot), cdn(self.w_pb_eta), cdn(self.w_pb_phi)]
            print("".join(str(key) + ": " + str(value) + ", " for key, value in zip(keys_w, vals_w)))
            print(f'scales: model_phi \n{cdn(self.model_phi.scales)}')
            print(f'scales: model_eta \n{cdn(self.model_eta.scales)}')

        # create and write csv for saving the loss in each epoch to plot the loss curve later
        if not exists(self.LOSS_PATH):
            write_csv_line(path=self.LOSS_PATH, line=keys)
        write_csv_line(path=self.LOSS_PATH, line=vals)

        # check if best loss improved and save model
        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.save_model(path=self.MODEL_PATH)

        # increase epochs counter and start time for next epoch
        self.epoch += 1
        self.start_time = time.time()

    def load_model(self, path: str):
        """
        function loads model, epochs and optimizer if a pth-file of previously trained model with the same name exists at path location
        :param path: path to load the pth-file from
        """
        ll = torch.load(path)
        self.model_eta.load_state_dict(ll['net_eta'])
        self.model_phi.load_state_dict(ll['net_phi'])
        self.model_eta_HC.load_state_dict(ll['net_eta_HC'])
        self.model_distance_fun.load_state_dict(ll['net_dist_fun'])
        self.epoch = ll['epoch']
        self.optimizerAdam.load_state_dict(ll['optim_Adam'])
        self.optimizerLBFGS.load_state_dict(ll['optim_LBFGS'])
        print(f'\n\n\nLoaded model from epoch {self.epoch}: \n\n\n')

    def save_model(self, path: str):
        """
        function saves a model, epochs and optimizer as a pth-file at the path
        :param path: path to save the pth-file from
        """
        torch.save(
            {'net_eta': self.model_eta.state_dict(),
             'net_phi': self.model_phi.state_dict(),
             'net_eta_HC': self.model_eta_HC.state_dict(),
             'net_dist_fun': self.model_distance_fun.state_dict(),
             'epoch': self.epoch,
             'optim_Adam': self.optimizerAdam.state_dict(),
             'optim_LBFGS': self.optimizerLBFGS.state_dict()},
            path)
        print(f'model checkpoint saved')

    def inference_eta_phi(self, x: np.ndarray, t: np.ndarray, z: np.array, batch_size=1000):
        """
        function makes an inference for surface elevation and velocity potential after the model is trained
        :param x: array of x values
        :param t: array of corresponding t values
        :param z: array of corresponding z values
        :param batch_size: batch size for the test loader
        :return: elevation eta_i and potential phi_i for all points (x_i, t_i, z_i)
       """

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tfn(x.astype(np.float32)), tfn(t.astype(np.float32)), tfn(z.astype(np.float32))),
                                                  batch_size=batch_size, shuffle=False)

        # initialize return tensors
        phi = np.zeros_like(x)
        eta = np.zeros_like(x)

        with torch.no_grad():
            for i, (x, t, z) in enumerate(test_loader):

                # eta inference for batch transformed to numpy and stored in return tensor
                eta_batch, _, _ = self.net_eta(x, t, cond='infer')
                eta[i * batch_size:(i + 1) * batch_size] = cdn(eta_batch)

                # phi inference for batch transformed to numpy and stored in return tensor
                phi_batch, _, _, _, _, _ = self.net_phi(x, t, z, cond='infer')
                phi[i * batch_size:(i + 1) * batch_size] = cdn(phi_batch)

        return eta, phi



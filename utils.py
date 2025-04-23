import torch
torch.set_default_dtype(torch.float32)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from libs.SSP import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.gridspec as gridspec

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')
plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', titlesize=7)
plt.rc('legend', title_fontsize=6)
plt.rc('legend', fontsize=6)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

dyn_cyan = '#1b7491'
dyn_red = '#8d1f22'
dyn_pink = '#BC91B9'
dyn_grey = '#5b7382'
dyn_dark = '#0c333f'
dyn_gras = '#1b9173'


def integral_trapz(y):
    """
    trapezoidal rule for uniformly spaced samples to approximate a definite integral numerically
    :param y: 1D array of function values
    :return: approximated integral under function graph
    """
    return (y[0] + y[-1]) / 2.0 + np.sum(y[1:-1])


def integral_trapz_2d(y):
    """
    2D trapezoidal rule for uniformly spaced grid in both axes to approximate a volume under a surface numerically
    :param y: 2D array of function values
    :return: approximated double integral value
    """
    # corner terms
    tr1 = (y[0, 0] + y[0, -1] + y[-1, 0] + y[-1, -1]) / 4.0

    # edge terms
    tr2 = (np.sum(y[0, 1:-1]) + np.sum(y[-1, 1:-1]) + np.sum(y[1:-1, 0]) + np.sum(y[1:-1, -1])) / 2.0

    # interior points
    tr3 = np.sum(y[1:-1, 1:-1])

    return tr1 + tr2 + tr3


def SSP(y_true, y_pred):
    """
    calculates the Surface Similarity Parameter (https://doi.org/10.1016/j.neunet.2022.09.023) between two signals (1D surfaces)
    :param y_true: 1D array of true surface elevation
    :param y_pred: 1D array of predicted surface elevation
    :return: scalar SSP value between 0 (perfect agreement) and 1 (no similarity)
    """
    # compute Fourier spectra
    spec1 = np.fft.fft(y_true)
    spec2 = np.fft.fft(y_pred)

    # calculate L2 norms
    nominator = np.sqrt(integral_trapz(np.square(np.abs(spec1 - spec2))))
    denominator = np.sqrt(integral_trapz(np.square(np.abs(spec1)))) + np.sqrt(integral_trapz(np.square(np.abs(spec2))))

    # normalized error
    SSP = np.divide(nominator, denominator)

    return SSP


def SSP_2D (y_true, y_pred):
    """
    calculates the 2D Surface Similarity Parameter (https://doi.org/10.1016/j.neunet.2022.09.023) between two surfaces
    :param y_true: 2D array of true surface elevation field
    :param y_pred: 2D array of predicted surface elevation field
    :return: scalar SSP value between 0 (perfect agreement) and 1 (no similarity)
    """
    # compute 2D Fourier spectra
    spec1 = np.fft.fft2(y_true)
    spec2 = np.fft.fft2(y_pred)

    # calculate L2 norms using 2D integration
    nominator = np.sqrt(integral_trapz_2d(np.square(np.abs(spec1 - spec2))))
    denominator = np.sqrt(integral_trapz_2d(np.square(np.abs(spec1)))) + np.sqrt(
        integral_trapz_2d(np.square(np.abs(spec2))))

    # normalized error
    SSP = np.divide(nominator, denominator)

    return SSP


def shift_phi_to_match_mean(phi, exact_phi):
    """
    Aligns two signals by matching their mean values through a constant shift.
    Useful for comparing signals where absolute offsets are not meaningful.
    :param phi: input signal to be shifted (1D or 2D array)
    :param exact_phi: reference signal whose mean will be matched
    :return: shifted version of `phi` with mean matching `exact_phi`

    """
    # Calculate the mean of phi and exact_phi
    phi_mean = np.mean(phi)
    target_mean = np.mean(exact_phi)

    # Calculate the mean difference
    mean_offset = target_mean - phi_mean

    # Add the mean difference to phi
    phi_shifted = phi + mean_offset

    return phi_shifted


def plotting_PINN_elevation(x, t, eta_true, eta_pred, xt_train, x_is, path_save, epoch='end', format_save='pdf'):
    """
    Visualize PINN surface elevation and ground truth with error metrics and cross-sections.
    :param x: spatial coordinates (1D array)
    :param t: temporal coordinates (1D array)
    :param eta_true: ground truth surface elevation (nt × nx) on z=0 from analytic solution
    :param eta_pred: reconstructed surface elevation (nt × nx) on z=0 from PINN
    :param x_is: x-indices for cross-section plots
    :param path_save: path to save the output figure
    :param epoch: epoch identifier for filename
    :param format_save: image format ('pdf', 'png', etc.)
    """
    # calculate global error metrics
    SSPs = np.zeros_like(x_is, dtype=float)
    MSEs = np.zeros_like(x_is, dtype=float)
    SSP_total = np.round(SSP_2D(eta_true, eta_pred), 3)
    MSE_total = np.round(np.mean(np.square(eta_true- eta_pred)), 8)

    # calculate location-specific errors for cross-sections
    for i, x_i in enumerate(x_is):
        SSPs[i] = SSP(eta_true[:, x_i], eta_pred[:, x_i])
        MSEs[i] = MSE(eta_true[:, x_i], eta_pred[:, x_i])

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw={'width_ratios': [10, 9], 'hspace': 0.4})
    # true surface elevation
    mima = np.max([np.nanmax(np.abs(eta_true))])
    h = axs[0,0].imshow(eta_true.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto',
                  vmin=-mima-0.1*mima, vmax=mima+0.1*mima)
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    axs[0,0].plot(xt_train[:, 1], xt_train[:, 0], 'x', label='measurements $\eta_\mathrm{m}(x=x_{\mathrm{wb},j},t)$' , markersize=5, c=dyn_red,
            clip_on=False)
    len_x = np.linspace(np.min(x)+0.005*np.max(x), np.max(x)-0.005*np.max(x), 5)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[0], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[1], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[2], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[3], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[4], '--', c='k', linewidth=0.9)
    axs[0,0].set_xlabel('$t \, [\mathrm{s}]$', fontsize=8)
    axs[0,0].set_ylabel('$x \, [\mathrm{m}]$', fontsize=8)
    axs[0,0].legend(frameon=True, loc='upper left', fontsize=6)
    axs[0,0].set_title('true elevation $\eta_\mathrm{true}(x,t)$', fontsize=7)

    # PINN surface elevation
    h = axs[1, 0].imshow(eta_pred.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto',
                  vmin=-mima-0.1*mima, vmax=mima+0.1*mima)
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[0], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[1], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[2], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[3], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[4], '--', c='k', linewidth=0.9)
    axs[1, 0].set_xlabel('$t \, [\mathrm{s}]$', fontsize=8)
    axs[1, 0].set_ylabel('$x \, [\mathrm{m}]$', fontsize=8)
    axs[1, 0].set_title(r'PINN solution $\Tilde{\eta}(x,t)$'+ f': SSP $={SSP_total}$, MSE $={MSE_total:.2e}$', fontsize=7)

    # cross section comparison
    subs = len(x_is)
    gs1 = gridspec.GridSpec(subs, 1)
    gs1.update(top=0.88, bottom=0.1, left=0.65, right=1, wspace=0.3, hspace=0.175)

    for i, (x_i, SSP_i, MSE_i) in enumerate(zip(np.flip(x_is), np.flip(SSPs), np.flip(MSEs))):
        if x_i <= np.size(x):
            ax = plt.subplot(gs1[i])
            ax.plot(t, eta_true[:, x_i], c=dyn_cyan, linestyle='-', linewidth=0.7, label='$\eta_\mathrm{true}(x,t)$')
            ax.plot(t, eta_pred[:, x_i], c=dyn_red, linestyle='--', linewidth=0.8, label='PINN $\Tilde{\eta}(x,t)$')
            ax.text(np.min(t)+0.03*(np.max(t)-np.min(t)), -mima-0.35*mima, '$  x = \,$' + f'{np.round(np.squeeze(x[x_i]),4)}' + '$\, \mathrm{m}$: ' + f'SSP $= {np.round(SSP_i, 3)}$, MSE $= {MSE_i:.2e}$',
                    fontsize=7)
            ax.set_xlim([np.min(t), np.max(t)])
            ax.set_ylim([-mima-0.45*mima, mima+0.2*mima])
            ax.tick_params(axis='both', labelsize=8)

            if i < subs-1:
                ax.get_xaxis().set_ticklabels([])
            else:
                ax.set_xlabel('$t \, [\mathrm{s}]$', fontsize=8)

            if i == 0:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=5, frameon=True, fontsize=7)
            if i == 2:
                ax.set_ylabel('elevation $\eta \, [\mathrm{m}]$')

    plt.savefig(f'{path_save}reconstruction_epoch_{epoch}.{format_save}', bbox_inches='tight', pad_inches=0.1)


def plotting_PINN_potential(x, t, phi_true, phi_pred, x_is, path_save, epoch='end', format_save='pdf'):
    """
    Visualize PINN velocity potential and ground truth with error metrics and cross-sections.
    :param x: spatial coordinates (1D array)
    :param t: temporal coordinates (1D array)
    :param phi_true: ground truth potential field (nt × nx) on z=0 from analytic solution
    :param phi_pred: reconstructed potential field (nt × nx) on z=0 from PINN
    :param x_is: x-indices for cross-section plots
    :param path_save: path to save the output figure
    :param epoch: epoch identifier for filename
    :param format_save: image format ('pdf', 'png', etc.)
    """

    # calculate global error metrics
    SSPs = np.zeros_like(x_is, dtype=float)
    MSEs = np.zeros_like(x_is, dtype=float)
    SSP_total = np.round(SSP_2D(phi_true, phi_pred), 3)
    MSE_total = np.round(np.mean(np.square(phi_true- phi_pred)), 5)

    # calculate location-specific errors for cross-sections
    for i, x_i in enumerate(x_is):
        SSPs[i] = SSP(phi_true[:, x_i], phi_pred[:, x_i])
        MSEs[i] = MSE(phi_true[:, x_i], phi_pred[:, x_i])

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw={'width_ratios': [10, 9], 'hspace': 0.4})
    # true potential field
    mima = np.max([np.nanmax(np.abs(phi_true))])
    h = axs[0,0].imshow(phi_true.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto',
                  vmin=-mima-0.2*mima, vmax=mima+0.2*mima)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    len_x = np.linspace(np.min(x)+0.005*np.max(x), np.max(x)-0.005*np.max(x), 5)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[0], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[1], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[2], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[3], '--', c='k', linewidth=0.9)
    axs[0, 0].plot(t, np.ones_like(t)*len_x[4], '--', c='k', linewidth=0.9)
    axs[0, 0].set_xlabel('$t \, [\mathrm{s}]$', fontsize=8)
    axs[0, 0].set_ylabel('$x \, [\mathrm{m}]$', fontsize=8)
    axs[0, 0].set_title(r'true potential $\Phi_\mathrm{true}(x,t, z=0)$', fontsize=7)

    # PINN potential field
    h = axs[1,0].imshow(phi_pred.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto',
                  vmin=-mima-0.2*mima, vmax=mima+0.2*mima)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[0], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[1], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[2], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[3], '--', c='k', linewidth=0.9)
    axs[1, 0].plot(t, np.ones_like(t)*len_x[4], '--', c='k', linewidth=0.9)
    axs[1,0].set_xlabel('$t \, [\mathrm{s}]$', fontsize=8)
    axs[1,0].set_ylabel('$x \, [\mathrm{m}]$', fontsize=8)
    axs[1,0].set_title(r'PINN solution $\Tilde{\Phi}(x,t, z=0)$'+ f', SSP $={SSP_total}$, MSE $={MSE_total:.2e}$', fontsize=7)

    # cross section comparison
    subs = len(x_is)
    gs1 = gridspec.GridSpec(subs, 1)
    gs1.update(top=0.88, bottom=0.1, left=0.65, right=1, wspace=0.3, hspace=0.175)

    for i, (x_i, SSP_i, MSE_i) in enumerate(zip(np.flip(x_is), np.flip(SSPs), np.flip(MSEs))):
        if x_i <= np.size(x):
            ax = plt.subplot(gs1[i])
            ax.plot(t, phi_true[:, x_i], c=dyn_cyan, linestyle='-', linewidth=0.7, label=r'$\Phi_\mathrm{true}(x,t, z=0)$')
            ax.plot(t, phi_pred[:, x_i], c=dyn_red, linestyle='--', linewidth=0.8, label=r'PINN $\Tilde{\Phi}(x,t, z=0)$')
            ax.text(np.min(t)+0.03*(np.max(t)-np.min(t)), -mima-0.4*mima, '$  x = \,$' + f'{np.round(np.squeeze(x[x_i]),4)}' + '$\, \mathrm{m}$: ' + f'SSP $= {np.round(SSP_i, 3)}$, MSE $= {MSE_i:.2e}$',
                    fontsize=7)
            ax.set_xlim([np.min(t), np.max(t)])
            ax.set_ylim([-mima-0.5*mima, mima+0.2*mima])
            ax.tick_params(axis='both', labelsize=8)

            if i < subs - 1:
                ax.get_xaxis().set_ticklabels([])
            else:
                ax.set_xlabel('$t \, [\mathrm{s}]$', fontsize=8)

            if i == 0:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=5, frameon=True, fontsize=7)
            if i == 2:
                ax.set_ylabel(r'potential $\Tilde{\Phi} \, \left[\frac{\mathrm{m}^2}{\mathrm{s}}\right]$', labelpad=-3)

    plt.savefig(f'{path_save}potential_at_0_epoch_{epoch}.{format_save}', bbox_inches='tight', pad_inches=0.1)


def plotting_losscurve_PB(path_loss: str, path_save, xmax, ymax, ymin, format_save='pdf', figsize=(6, 5)):
    """
    Generate loss curve visualization for physics-informed training.
    :param path_loss: path to CSV file containing loss history
    :param path_save: path to save the output figure
    :param xmax: maximum x-axis value (epoch limit)
    :param ymax: maximum y-axis value (log scale)
    :param ymin: minimum y-axis value (log scale)
    :param format_save: output format of figure ('pdf', 'png', etc.)
    :param figsize: figure dimensions in inches (width, height)
    """
    df = pd.read_csv(path_loss)
    plt.figure(figsize=figsize)
    plt.plot(df.MSE_pde[1:], color=dyn_cyan, label='$\mathcal{L}_\mathrm{Lap}$', linewidth=0.9)
    plt.plot(df.MSE_bc_kin[1:], color=dyn_dark, label='$\mathcal{L}_\mathrm{BC,kin}$', linewidth=0.9)
    plt.plot(df.MSE_bc_dyn[1:], color=dyn_red, label='$\mathcal{L}_\mathrm{BC,dyn}$', linewidth=0.9)
    plt.plot(df.MSE_bc_bot[1:], color=dyn_gras, label='$\mathcal{L}_\mathrm{BC,bot}$', linewidth=0.9)
    plt.plot(df.MSE_pb_eta[1:], color='orange', label='$\mathcal{L}_\mathrm{PB,\eta}$', linewidth=0.9)
    plt.plot(df.MSE_pb_phi[1:], color='darkgoldenrod', label='$\mathcal{L}_\mathrm{PB,\Phi}$', linewidth=0.9)
    plt.plot(df.MSE_data[1:], '--', color=dyn_pink, label='$\mathrm{MSE_{data}}$', linewidth=0.9)
    plt.ylim([ymin, ymax])
    plt.xlim([0, xmax])
    plt.xlabel('epochs', fontsize=8)
    plt.ylabel('MSE', fontsize=8)
    plt.yscale('log')
    plt.grid()
    plt.tick_params(axis='both', labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{path_save}/loss.{format_save}')
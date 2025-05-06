import os
import numpy as np
# to avoid str(scalar) returning "np.float64(0.5)" rather than "0.5"
np.set_printoptions(legacy="1.25")
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import tools.cycle_utils as cu

import pysr
from pysr import PySRRegressor

size = 19
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": "Times",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": size,
    "axes.labelsize": size,
    "axes.titlesize": size,
    "figure.titlesize": size,
    "xtick.labelsize": size,
    "ytick.labelsize": size,
    "legend.fontsize": size,
})


def read_result_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()[1:]

    gammas = list()
    betas = list()

    for line in lines:
        gamma, beta = line.split("\t")[:2]
        gammas.append(float(gamma))
        betas.append(float(beta))

    return gammas, betas


def read_result_file_multistep(file_path):
    with open(file_path) as f:
        lines = f.readlines()[1:]

    all_gamma_intervals = list()
    betas = list()

    for line in lines:
        gamma_intervals_str, beta_str = line.split("\t")[:2]
        betas.append(float(beta_str))
        
        gamma_intervals_str = gamma_intervals_str.split(";")
        gamma_intervals = []
        for gamma_interval in gamma_intervals_str:
            I = gamma_interval.strip('()').split(', ')
            gamma_intervals += [ (float(I[0]), float(I[1])) ]
            
        all_gamma_intervals.append(gamma_intervals)

    return all_gamma_intervals, betas


def write_result_file(logdir, filename, gammas, betas):
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    file_path = os.path.join(logdir, filename)
    with open(file_path, "w") as f:
        f.write("gamma\tbeta\n")
        for gamma, beta in zip(gammas, betas):
            f.write("{}\t{}\n".format(gamma, beta))


def write_result_file_multistep(logdir, filename, gamma_intervals, betas):
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    file_path = os.path.join(logdir, filename)
    with open(file_path, "w") as f:
        f.write("gamma_intervals\tbeta\n")
        for gamma_intervals, beta in zip(gamma_intervals, betas):
            gamma_intervals_str = [str(I) for I in gamma_intervals]
            
            f.write("{}\t{}\n".format(";".join(gamma_intervals_str), beta))


def get_colored_graphics_HB_multistep_lyapunov(mu, L, max_lyapunov_steps, folder="results/"):
    method = "HB"
    figdir = os.path.join(folder, "figures")
    if not os.path.isdir(figdir):
        os.makedirs(figdir)
    
    fig_union, ax_union = plt.subplots(nrows=1, ncols=1, figsize=(15, 9), 
                                       constrained_layout=True)
    axs = []
    figs = []

    # Background
    betas = np.linspace(0, 1, 300 + 1, endpoint=False)[1:]

    # Lyapunov
    for T in range(1, max_lyapunov_steps + 1):
        fn = "{}_mu{:.2f}_L{:.0f}_steps_{:d}.txt".format(method, mu, L, T)
        result_path = os.path.join(folder, "lyapunov", fn)
        gamma_intervals_lyap, betas_lyap = read_result_file_multistep(file_path=result_path)
        
        x_green = list()
        y_green = list()
        for gamma_intervals, beta in zip(gamma_intervals_lyap, betas_lyap):
            for gamma_min, gamma_max in gamma_intervals:
                if gamma_max - gamma_min > .01:
                    x_green += list(np.linspace(gamma_min, gamma_max, 500))
                    y_green += [beta] * 500
                
        # add lyapunov
        ax_union.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")
        
        # now do a separate one where it's not unionized
        fig_T, ax_T = plt.subplots(nrows=1, ncols=1, figsize=(15, 9), 
                                   constrained_layout=True)
        
        # ======== also fit a curve to the boundary ========
        # if T == 1:
        #     b_gammas = []
        #     b_betas = []
        #     for gamma_intervals, beta in zip(gamma_intervals_lyap, betas_lyap):
        #         # just take the first entry T=1
        #         gamma_max = gamma_intervals[0][1]
        #         bound = cu.bound("HB", L, beta)
        #         if np.abs(gamma_max - bound) < 1e-2:
        #             # toss out the values that are capped at the boundary
        #             continue
        #         b_gammas.append(gamma_max)
        #         b_betas.append(beta)
            
        #     # print("mu", mu)
        #     # # print(np.min(b_betas))
        #     # coeffs = np.polyfit(b_betas[:100], b_gammas[:100], deg=2)
        #     # print(coeffs)
        #     # poly = np.poly1d(coeffs)

        #     def model(zz):
        #         return (zz - 1.0044357314273449)*(-np.exp(3.069555330084005*zz)-1.2140179016574235)

        #     yy = np.linspace(np.min(b_betas), 1, 500)
        #     xx = model(yy)
        #     # ind = np.argmin( np.abs(2*(1+yy) - xx) )
        #     # beta_cross = yy[ind]
        #     # print(beta_cross)
        #     # gamma_cross = xx[ind]
        #     # ax_T.scatter(gamma_cross, beta_cross, s=50, color="darkgreen", zorder=100)
        #     ax_T.plot(xx, yy, linewidth=2, color="darkgreen", zorder=100)
        # =======================================================
        
        axs += [ax_T]
        figs += [fig_T]
        ax_T.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")
        
    # ========== plot cycles manually for all axes, separate and union ==========
    valid_tunings = cu.get_cycle_tunings(mu, L, betas)
    for axq in axs + [ax_union]:
        for _, (K, valid_betas, valid_Lgammas) in valid_tunings.items():
            color = "red" if K.is_integer() else "orange"
            axq.scatter(valid_Lgammas, valid_betas, color=color, s=1, zorder=100)
            
        m = 100
        Lgammas = np.linspace(2, 4, num=m)
        beta_Q = Lgammas/2 - 1
        
        axq.plot(Lgammas, beta_Q, color="black")
        axq.plot(np.linspace(0,2,10), np.zeros(10), color="black")
        axq.axhline(1.0, color="black")
        axq.set_xlabel(r"$\gamma$")
        axq.set_ylabel(r"$\beta$", rotation=0, labelpad=10)
        
        # plot the union of all cycle boundaries including fractional K
        # theoretically computed
        kappa = mu/L
        
        omega = 8
        K_max = omega*10
        start = 2
        K_range = [i/omega for i in range(omega*start, K_max+1)]
        
        Phis = np.cos( 2*np.pi /  np.array(K_range) )
        Gammas = np.zeros_like(Phis)
        Betas = np.zeros_like(Phis)

        for i, phi in enumerate(Phis):
            def gamma(beta):
                return kappa*(beta**2 - 2*(2*phi-1)*beta + 1) / (kappa*beta + 1)
            
            def func(beta):
                # now this is the function we want to solve the roots for
                q4 = (2*kappa**2 - kappa)*beta**4
                q3 = (-4*kappa**2*phi**2 - 2*kappa**2 + 4*kappa - 2)*beta**3
                q2 = (8*phi*kappa**2 - 2*kappa**2 + 8*kappa*phi**2 - 16*kappa*phi + 2*kappa + 8*phi - 2)*beta**2
                q1 = (-4*phi**2 - 2*kappa**2 + 4*kappa - 2)*beta
                q0 = -kappa + 2
                
                return q4 + q3 + q2 + q1 + q0
            
            sol = root_scalar(func, bracket=[0,1], method="brentq")
            beta = sol.root
            Betas[i] = beta
            Gammas[i] = gamma(beta)
            
        axq.scatter(Gammas/mu, Betas, s=10, color="black", zorder=100)
        
        # also plot Ghadimi's region
        Betas = np.linspace(0, 1, 300 + 1, endpoint=False)[1:]
        Gammas = 2*(1-Betas**2) / (L - mu*Betas)
        axq.scatter(Gammas, Betas, s=10, color="darkgreen", zorder=100)

    # ========================================
    
    figfn = "{}_mu{:.2f}_L{:.0f}_steps{}_union_colored.png".format(method, mu, L, max_lyapunov_steps)
    fig_union.savefig(os.path.join(figdir, figfn), bbox_inches="tight")
    
    for T in range(1, max_lyapunov_steps + 1):
        fig = figs[T-1]
        figfn = "{}_mu{:.2f}_L{:.0f}_steps{}_single_colored.png".format(method, mu, L, T)
        fig.savefig(os.path.join(figdir, figfn), bbox_inches="tight")


def get_colored_graphics(method, mu, L, max_cycle_length, add_background=True, add_lyapunov=True, inplot=True, folder="results/"):
    fig = plt.figure(figsize=(15, 9))
    plt.xlabel(r"$\gamma$")
    if method == "GD":
        plt.ylabel(r"$\varepsilon$", rotation=0, labelpad=10)
    else:
        plt.ylabel(r"$\beta$", rotation=0, labelpad=10)

    ax = plt.subplot(111)

    if method == "HB" and inplot:
        axins = zoomed_inset_axes(ax, zoom=3.5, loc="lower right")

    # Background
    x_grey = list()
    y_grey = list()
    if method == "TOS":
        betas = np.linspace(0, 2, 300 + 1, endpoint=False)[1:]
    else:
        betas = np.linspace(0, 1, 300 + 1, endpoint=False)[1:]
    for beta in betas:
        x_grey += list(np.linspace(0, cu.bound(method=method, L=L, beta=beta), 500))
        y_grey += [beta] * 500
    if add_background:
        ax.plot(x_grey, y_grey, '.', color="gainsboro", label="no conclusion")

    if method == "HB" and inplot:
        axins.plot(x_grey, y_grey, ".", color="gainsboro")

    legends = ["no conclusion"]
    colors = ["gainsboro"]

    # Lyapunov
    gammas_lyap, betas_lyap = read_result_file(
        file_path=folder + "lyapunov/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))
    x_green = list()
    y_green = list()
    for gamma_max, beta in zip(gammas_lyap, betas_lyap):
        if gamma_max > .01:
            x_green += list(np.linspace(0, gamma_max, 500))
            y_green += [beta] * 500
    if add_lyapunov:
        ax.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")

    if method == "HB" and inplot:
        axins.plot(x_green, y_green, ".", color="yellowgreen")

    legends.append("convergence")
    colors.append("yellowgreen")

    # Cycles
    color_map = plt.get_cmap('YlOrRd')
    for K in range(max_cycle_length, 1, -1):
        try:
            gammas_cycle, betas_cycle = read_result_file(
                file_path=folder + "cycles/{}_mu{:.2f}_L{:.0f}_K{:.0f}.txt".format(method, mu, L, K))
            x_red = list()
            y_red = list()
            for gamma_min, beta in zip(gammas_cycle, betas_cycle):
                if gamma_min <= cu.bound(method, L, beta):
                    x_red += list(np.linspace(gamma_min, cu.bound(method, L, beta), 500))
                    y_red += [beta] * 500
            color_scale = (max_cycle_length + 1 - K) / (max_cycle_length - 1)
            color = color_map(color_scale)

            ax.plot(x_red, y_red, '.', color=color, label="cycle of length {}".format(K))
            if method == "HB" and inplot:
                axins.plot(x_red, y_red, ".", color=color)
            legends.append("cycle of length {}".format(K))
            colors.append(color)
        except FileNotFoundError:
            pass
    if method == "HB" and inplot:
        x1 = -0.01
        x2 = 0.25
        y1 = 0.95
        y2 = 1.002
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.7")
        ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linewidth=0.6, color="grey")

    bounds = list(range(len(colors[2:])))
    norm = mpl.colors.BoundaryNorm(np.array(bounds) + 2, len(colors[2:]))
    position = fig.add_axes([0.42, 0.92, 0.46, 0.02])  # [x_init, y_init, width, height]
    clbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=mpl.colors.ListedColormap(colors[2:][::-1])),
                         cax=position, orientation="horizontal", shrink=0.8, fraction=0.1, aspect=50)
    clbar.ax.set_title("Length of the shortest cycle")

    handles = [mlines.Line2D([], [], color=colors[1], marker="s",
                             linestyle="None", markersize=12, markeredgecolor="black")]
    labels = [""]
    fig.legend(handles, labels, bbox_to_anchor=(0.37, 1), title_fontsize=20, title="Convergence", frameon=False)

    handles = [mlines.Line2D([], [], color=colors[0], marker="s",
                             linestyle="None", markersize=12, markeredgecolor="black")]
    labels = [""]
    fig.legend(handles, labels, bbox_to_anchor=(0.25, 1), title_fontsize=20, title="No conclusion", frameon=False)

    figdir = os.path.join(folder, "figures")
    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    figname = "{}_mu{:.2f}_L{:.0f}_colored.png".format(method, mu, L)
    plt.savefig(os.path.join(figdir, figname), bbox_inches="tight")


if __name__ == "__main__":
    for method in ["HB", "NAG", "GD", "TOS"]:
        for mu in [0]:
            try:
                get_colored_graphics(method=method, mu=mu, L=1, max_cycle_length=25, folder="../results/")
            except FileNotFoundError:
                pass

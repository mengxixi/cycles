import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

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


def write_result_file(logdir, filename, gammas, betas):
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    file_path = os.path.join(logdir, filename)
    with open(file_path, "w") as f:
        f.write("gamma\tbeta\n")
        for gamma, beta in zip(gammas, betas):
            f.write("{}\t{}\n".format(gamma, beta))


def bound(method, L, beta):
    if method == "HB":
        return 2 * (1 + beta) / L
    elif method == "NAG":
        return (1 + 1 / (1 + 2 * beta)) / L
    elif method == "GD":
        return 2 / L
    elif method == "TOS":
        return 2 / L
    else:
        raise Exception


def is_valid(gamma, beta, L):
    return gamma*L < 2 * (1+beta) and gamma*L >= 0 and gamma*L <=4


def valid_for_another_K(beta, gamma, K, mu, kappa):
    # return False

    theta = 2*np.pi/K
    cos = np.cos(theta)
    
    a = mu**2
    b = - 2*mu * (beta - cos + kappa * (1-beta*cos))
    c = 2*kappa*(1-cos)*(1+beta**2-2*beta*cos)
    
    return a * gamma**2 + b * gamma + c <= 0


def get_cycle_tunings(mu, L, betas):
    kappa = mu/L
    omega = 1 # change to 2 to get multiples of fractions
    K_max = omega*10
    start = 3
    K_range = [i/omega for i in range(omega*start, K_max)]
    valid_tunings = {}
    
    for i, K in enumerate(K_range):

        valid_betas = []
        valid_Lgammas = []

        theta = 2*np.pi/K
        cos = np.cos(theta)
        
        for beta in betas:
            a = mu**2
            b = - 2*mu * (beta - cos + kappa * (1-beta*cos))
            c = 2*kappa*(1-cos)*(1+beta**2-2*beta*cos)
            
            if b**2 < 4*a*c:
                continue
            
            gamma1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            gamma2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            
            v1 = is_valid(gamma1, beta, L)
            v2 = is_valid(gamma2, beta, L)
            
            if v1:
                included = False
                for k in K_range[:i]:
                    included = included or valid_for_another_K(beta, gamma1, k, mu, kappa)
                
                if not included:
                    valid_betas.append(beta)
                    valid_Lgammas.append(gamma1*L)
            
            if v2:
                included = False
                for k in K_range[:i]:
                    included = included or valid_for_another_K(beta, gamma2, k, mu, kappa)
                
                if not included:
                    valid_betas.append(beta)
                    valid_Lgammas.append(gamma2*L)

        valid_tunings[str(K)] = (K, valid_betas, valid_Lgammas)
        
    return valid_tunings


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
        gammas_lyap, betas_lyap = read_result_file(file_path=result_path)
        
        x_green = list()
        y_green = list()
        for gamma_max, beta in zip(gammas_lyap, betas_lyap):
            if gamma_max > .01:
                x_green += list(np.linspace(0, gamma_max, 500))
                y_green += [beta] * 500
                
        # add lyapunov
        ax_union.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")
        
        # now do a separate one where it's not 
        fig_T, ax_T = plt.subplots(nrows=1, ncols=1, figsize=(15, 9), 
                                   constrained_layout=True)
        axs += [ax_T]
        figs += [fig_T]
        ax_T.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")
        
    # ========== plot cycles manually for all axes, separate and union ==========
    valid_tunings = get_cycle_tunings(mu, L, betas)
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
        x_grey += list(np.linspace(0, bound(method=method, L=L, beta=beta), 500))
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
                if gamma_min <= bound(method, L, beta):
                    x_red += list(np.linspace(gamma_min, bound(method, L, beta), 500))
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
    plt.savefig(figname, bbox_inches="tight")


if __name__ == "__main__":
    for method in ["HB", "NAG", "GD", "TOS"]:
        for mu in [0]:
            try:
                get_colored_graphics(method=method, mu=mu, L=1, max_cycle_length=25, folder="../results/")
            except FileNotFoundError:
                pass

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
# import pysr
# from pysr import PySRRegressor

import algorithms.heavy_ball.lyapunov as hblyap
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction

import cvxpy as cp

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


def get_colored_graphics_HB_lyapunov_all_history(mu, L, max_lyapunov_steps, folder="results/"):
    method = "HB"
    figdir = os.path.join(folder, "figures")
    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    axs = []
    figs = []

    # Background
    betas = np.linspace(0, 1, 300 + 1, endpoint=False)[1:]

    # Lyapunov
    T = max_lyapunov_steps
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
            
    # now do a separate one where it's not unionized
    fig_T, ax_T = plt.subplots(nrows=1, ncols=1, figsize=(15, 9), 
                                constrained_layout=True)
    ax_T.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")
    
    # ========== plot cycles manually for all axes, separate and union ==========
    valid_tunings = cu.get_cycle_tunings(mu, L, betas)
    for _, (K, valid_betas, valid_Lgammas) in valid_tunings.items():
        color = "red" if K.is_integer() else "orange"
        ax_T.scatter(valid_Lgammas, valid_betas, color=color, s=1, zorder=100)
        
    m = 100
    Lgammas = np.linspace(2, 4, num=m)
    beta_Q = Lgammas/2 - 1
    
    ax_T.plot(Lgammas, beta_Q, color="black")
    ax_T.plot(np.linspace(0,2,10), np.zeros(10), color="black")
    ax_T.axhline(1.0, color="black")
    ax_T.set_xlabel(r"$\gamma$")
    ax_T.set_ylabel(r"$\beta$", rotation=0, labelpad=10)
    
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
        
    ax_T.scatter(Gammas/mu, Betas, s=10, color="black", zorder=100)
    
    # also plot Ghadimi's region
    Betas = np.linspace(0, 1, 300 + 1, endpoint=False)[1:]
    Gammas = np.minimum(2*(1-Betas**2) / (L - mu*Betas), 2/L)
    ax_T.scatter(Gammas, Betas, s=10, color="darkgreen", zorder=100)

    # ========================================
    figfn = "{}_mu{:.2f}_L{:.0f}_steps{}_all_history_colored.png".format(method, mu, L, max_lyapunov_steps)
    fig_T.savefig(os.path.join(figdir, figfn), bbox_inches="tight")
    

def get_colored_graphics_HB_multistep_lyapunov(mu, L, max_lyapunov_steps, pep_check=True, folder="results/"):
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
        # to keep track of points that are also verifiable by PEPit
        x_pepit = list()
        y_pepit = list()
        for gamma_intervals, beta in zip(gamma_intervals_lyap, betas_lyap):
            for gamma_min, gamma_max in gamma_intervals:
                if gamma_max - gamma_min > .01:
                    x_green += list(np.linspace(gamma_min, gamma_max, 500))
                    y_green += [beta] * 500
                    
                    # verify via pepit at (gamma_max, beta)
                    if pep_check:
                        pep_ok = verify_multistep_lyapunov_via_pepit(gamma_max, beta, mu, L, T)
                        if pep_ok:
                            x_pepit += list(np.linspace(gamma_min, gamma_max, 500))
                            y_pepit += [beta] * 500
                
        # add lyapunov
        ax_union.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")
        ax_union.plot(x_pepit, y_pepit, '.', color="deeppink", label="convergence")
        
        # now do a separate one where it's not unionized
        fig_T, ax_T = plt.subplots(nrows=1, ncols=1, figsize=(15, 9), 
                                   constrained_layout=True)
        axs += [ax_T]
        figs += [fig_T]
        ax_T.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")
        ax_T.plot(x_pepit, y_pepit, '.', color="deeppink", label="convergence")
        
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
        Gammas = np.minimum(2*(1-Betas**2) / (L - mu*Betas), 2/L)
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


def verify_multistep_lyapunov_via_pepit(gamma, beta, mu, L, T):
    rho = 1 
    value, _, P, p, _, _ = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta, gamma, mu, L, rho, T, return_all=True)
    assert value == 0.
    
    # extract the values for corresponding terms in the lyapunov function
    Pmat = P.value
    pvec = p.value
    
    a = Pmat[0,0]
    b = Pmat[2,2]
    c = Pmat[3,3]
    d = Pmat[1,2]
    e = Pmat[1,3]
    f = Pmat[2,3]
    
    p0 = pvec[0]
    p1 = pvec[1]
    
    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding function value f0
    x0 = problem.set_initial_point()
    x1 = problem.set_initial_point()
    f0 = func(x0)
    g0, f0 = func.oracle(x0)
    g1, f1 = func.oracle(x1)
    
    V0 = a*(x1-x0)**2 + 2*d*(x1-x0)*g0 + 2*e*(x1-x0)*g1 + 2*f*g1*g0 + b*g0**2 + c*g1**2 \
        + p0*(f0-fs) + p1*(f1-fs)
    
    x_previous = x0
    x_current = x1
    for _ in range(T):
      x_next = x_current + beta*(x_current-x_previous) - gamma * func.gradient(x_current)
      x_previous = x_current
      x_current = x_next
    
    xk = x_previous
    xkk = x_current
    gk, fk = func.oracle(xk)
    gkk, fkk = func.oracle(xkk)  
    
    VT = a*(xkk-xk)**2 + 2*d*(xkk-xk)*gk + 2*e*(xkk-xk)*gkk + 2*f*gkk*gk + b*gk**2 + c*gkk**2 \
        + p0*(fk-fs) + p1*(fkk-fs)

    # Set the initial constraint that is the distance between f(x0) and f(x^*)
    problem.set_initial_condition(V0 <= 1)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(VT)
    
    # Solve the PEP
    try: # Try to solve the problem, handle exceptions if they occur
        pepit_tau = problem.solve(verbose=0)
    except cp.error.SolverError: # Catch the specific ValueError
        print(f"Solver failed for gamma={gamma}, beta={beta}. Returning inf")
        return False # Return inf for both rates in case of failure
    if pepit_tau is None:
      # Idk why it sometimes returns None
      return False
  
    return pepit_tau <= 1.


if __name__ == "__main__":
    for method in ["HB", "NAG", "GD", "TOS"]:
        for mu in [0]:
            try:
                get_colored_graphics(method=method, mu=mu, L=1, max_cycle_length=25, folder="../results/")
            except FileNotFoundError:
                pass

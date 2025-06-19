import os
import argparse
from math import inf

import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib as mpl

import algorithms.heavy_ball.lyapunov as hblyap

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": "Helvetica",
    "figure.dpi": 600
})

TMP_DIR = "tmp"


def get_gamma_beta_pair(mu, L, K):
    kappa = mu/L
    phi = np.cos( 2*np.pi /  K )
    
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

    return gamma(beta)/mu, beta


def bisection_max_beta(beta_start, gamma, mu, L, rho, lyapunov_steps):
    beta_min = beta_start
    beta_max = 1.0
    while beta_max - beta_min > 1e-12:
        beta_next = (beta_max + beta_min) / 2
        value = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta_next, gamma, mu, L, rho, lyapunov_steps)

        if value != inf:
            beta_min = beta_next
        else:
            beta_max = beta_next
    
    return beta_min


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', '--K_list', nargs='+', type=float)
                            
    args = parser.parse_args()
    
    # setup
    T = 1 # number of lyapunov steps to run
    rho = 1
    
    # fix L = 1 for now, we'll study its dependency later
    L = 1

    # fix mu
    K_list = args.K_list
    nrows = 8
    ncols = 1
    fig_all, axs_all = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(2.5*ncols,2.5*nrows),
                            constrained_layout=True)
    colors = mpl.cm.viridis(np.linspace(0,1,len(K_list)))
    for i_K, K in enumerate(K_list):
        n_pts = 100
        mu_list = np.linspace(0, 1, n_pts+2, endpoint=False)[1:-1]

        a_list = np.zeros_like(mu_list)
        b_list = np.zeros_like(mu_list)
        c_list = np.zeros_like(mu_list)
        d_list = np.zeros_like(mu_list)
        e_list = np.zeros_like(mu_list)
        f_list = np.zeros_like(mu_list)
        p1_list = np.zeros_like(mu_list)
        p2_list = np.zeros_like(mu_list)

        for i, mu in enumerate(mu_list):
            gamma, beta = get_gamma_beta_pair(mu, L, K)
            if T > 1:
                beta = bisection_max_beta(beta, gamma, mu, L, rho, T)
            
            value, _, P, p, _, _ = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, T, return_all=True)
            assert value == 0.

            a = P.value[0,0]
            b = P.value[2,2]
            c = P.value[3,3]
            d = P.value[1,2]
            e = P.value[1,3]
            f = P.value[2,3]
            p1 = p.value[0]
            p2 = p.value[1]
            
            a_list[i] = a
            b_list[i] = b
            c_list[i] = c
            d_list[i] = d
            e_list[i] = e
            f_list[i] = f
            p1_list[i] = p1
            p2_list[i] = p2

        # make plots
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(2.5*ncols,2.5*nrows),
                                constrained_layout=True)


        for j, (params, label) in enumerate(zip([a_list, b_list, c_list, d_list, e_list, f_list, p1_list, p2_list], ["a", "b", "c", "d", "e", "f", r"$p_1$", r"$p_2$"])):
            ax = axs[j]
            ax.plot(mu_list, params, linewidth=2, color=colors[i_K])
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            # if label == "d":
            #     coeffs = np.polyfit(mu_list, params, deg=1)
            #     print("K=%d"%K, coeffs)

            if j == nrows - 1:
                ax.set_xlabel(r"$\mu$", fontsize=17)
                
            ax = axs_all[j]
            lab = r"$K=%.2f$" % K
            ax.plot(mu_list, params, linewidth=2, color=colors[i_K], label=lab)
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            if j == nrows - 1:
                ax.set_xlabel(r"$\mu$", fontsize=17)

        fig.suptitle(r"$K=%.2f$" % K, fontsize=17)

        # save
        figname = "lyap_param_vs_mu_K=%.2f.png" % K
        fig_fn = os.path.join(TMP_DIR, figname)
        fig.savefig(fig_fn)
        # plt.savefig(fig_fn.replace("png", "pdf"))
        print("Figure saved at \n%s" % fig_fn)

    # save
    axs_all[2].legend(frameon=False)
    figname = "lyap_param_vs_mu_all.png"
    fig_fn = os.path.join(TMP_DIR, figname)
    fig_all.savefig(fig_fn)
    # plt.savefig(fig_fn.replace("png", "pdf"))
    print("Figure saved at \n%s" % fig_fn)

import os
import argparse

import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

import algorithms.heavy_ball.lyapunov as hblyap

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": "Helvetica",
    "figure.dpi": 600
})

TMP_DIR = "tmp"


def get_gamma_beta_pair(kappa, K):
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

    return gamma(beta), beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", "--K", type=str)
    
    args = parser.parse_args()
    
    # setup
    T = 1 # number of lyapunov steps to run
    rho = 1
    
    # fix L = 1 for now, we'll study its dependency later
    L = 1

    # fix K>=2, which uniquely determines a pair (gamma, beta) on the boundary
    # for a given mu;
    # this pair should be verifiable by the lyapunov function we enforce
    K = float(args.K)

    n_pts = 100
    mu_list = np.linspace(0, 1, n_pts+1, endpoint=False)[1:]

    a_list = np.zeros_like(mu_list)
    b_list = np.zeros_like(mu_list)
    c_list = np.zeros_like(mu_list)
    d_list = np.zeros_like(mu_list)

    for i, mu in enumerate(mu_list):
        kappa = mu/L
        gamma, beta = get_gamma_beta_pair(kappa, K)
        
        value, _, P, p, _, _ = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, T, return_all=True)
        assert value == 0.

        b = P.value[0,3]
        a = P.value[0,0]
        c = P.value[3,3]
        d = p.value[1]
        
        a_list[i] = a
        b_list[i] = b
        c_list[i] = c
        d_list[i] = d

    # make plots
    nrows = 4
    ncols = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(2.5*ncols,2.5*nrows),
                            constrained_layout=True)


    for j, (params, label) in enumerate(zip([a_list, b_list, c_list, d_list], ["a", "b", "c", "d"])):
        ax = axs[j]
        ax.plot(mu_list, params, linewidth=1)
        ax.set_ylabel(r"$%s$" % label, fontsize=17)    
        
        if j == nrows - 1:
            ax.set_xlabel(r"$\mu$", fontsize=17)

    fig.suptitle(r"$K=%.2f$" % K, fontsize=17)

    # save
    figname = "lyap_param_vs_mu_K=%.2f.png" % K
    fig_fn = os.path.join(TMP_DIR, figname)
    plt.savefig(fig_fn)
    # plt.savefig(fig_fn.replace("png", "pdf"))
    print("Figure saved at \n%s" % fig_fn)

import os
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mu', '--mu_list', nargs='+', type=float)
                            
    args = parser.parse_args()
    
    # setup
    T = 1 # number of lyapunov steps to run
    rho = 1
    
    # fix L = 1 for now, we'll study its dependency later
    L = 1

    # fix mu
    mu_list = args.mu_list
    nrows = 4
    ncols = 2
    fig_all, axs_all = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(2.5*ncols,2.5*nrows),
                            constrained_layout=True)
    colors = mpl.cm.viridis(np.linspace(0,1,len(mu_list)))
    for i_mu, mu in enumerate(mu_list):
        n_pts = 100
        kappa = mu/L
        _, beta_min = get_gamma_beta_pair(mu, L, K=3)
        # beta_min = ( (2-kappa) - np.sqrt((1-kappa)*(5-kappa)) ) / (2*kappa - 1)
        Betas = np.linspace(beta_min, 1, num=100)
        Gammas = np.zeros_like(Betas)
        for ib, beta in enumerate(Betas):
            gamma = (1-beta)/((1-beta*kappa)**2)
            gamma *= (1+3*beta*(1-kappa)-kappa*beta**2 + np.sqrt(4*beta*(2-kappa-kappa*beta)*(1+beta-2*beta*kappa)) )
            Gammas[ib] = gamma

        a_list = np.zeros_like(Betas)
        b_list = np.zeros_like(Betas)
        c_list = np.zeros_like(Betas)
        d_list = np.zeros_like(Betas)

        for i, (gamma, beta) in enumerate(zip(Gammas, Betas)):            
            value, _, P, p, _, _ = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, T, return_all=True)
            assert value == 0.

            b = P.value[0,3]
            a = P.value[0,0]
            c = P.value[3,3]
            d = p.value[1]
            
            # mu_factor = (mu**1.5)/2.5
            a_list[i] = a #*mu_factor
            b_list[i] = b
            c_list[i] = c
            d_list[i] = d

        # make plots
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(2.5*ncols,2.5*nrows),
                                constrained_layout=True)


        for j, (params, label) in enumerate(zip([a_list, b_list, c_list, d_list], ["a", "b", "c", "d"])):
            ax = axs[j,0]
            ax.plot(Gammas, params, linewidth=2, color=colors[i_mu])
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            if j == nrows - 1:
                ax.set_xlabel(r"$\gamma$", fontsize=17)
                
            ax = axs_all[j, 0]
            ax.plot(Gammas, params, linewidth=2, color=colors[i_mu], label=r"$\mu=%.2f$" % mu)
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            if j == nrows - 1:
                ax.set_xlabel(r"$\gamma$", fontsize=17)
                
            ax = axs[j,1]
            ax.plot(Betas, params, linewidth=2, color=colors[i_mu])
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            if j == nrows - 1:
                ax.set_xlabel(r"$\beta$", fontsize=17)
                
            ax = axs_all[j, 1]
            ax.plot(Betas, params, linewidth=2, color=colors[i_mu], label=r"$\mu=%.2f$" % mu)
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            if j == nrows - 1:
                ax.set_xlabel(r"$\beta$", fontsize=17)

        fig.suptitle(r"$\mu=%.2f$" % mu, fontsize=17)

        # save
        figname = "lyap_param_vs_K_mu=%.2f.png" % mu
        fig_fn = os.path.join(TMP_DIR, figname)
        fig.savefig(fig_fn)
        # plt.savefig(fig_fn.replace("png", "pdf"))
        print("Figure saved at \n%s" % fig_fn)

    # save
    axs_all[2,0].legend(frameon=False)
    axs_all[2,1].legend(frameon=False)
    figname = "lyap_param_vs_K_all.png"
    fig_fn = os.path.join(TMP_DIR, figname)
    fig_all.savefig(fig_fn)
    # plt.savefig(fig_fn.replace("png", "pdf"))
    print("Figure saved at \n%s" % fig_fn)

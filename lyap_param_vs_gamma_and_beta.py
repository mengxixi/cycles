import os
import pickle
import itertools
import argparse

from math import inf
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import algorithms.heavy_ball.lyapunov as hblyap

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": "Helvetica",
    "figure.dpi": 600
})
CONTOUR_CMAP = "jet"

TMP_DIR = "tmp"
LOG_DIR = "logs"


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
    # parser.add_argument('-mu', '--mu_list', nargs='+', type=float)
    parser.add_argument('-mu', '--mu_list', nargs='+', type=float)
    parser.add_argument('-K', '--K_list', nargs='+', type=float)
                            
    args = parser.parse_args()
    
    # setup
    T = 1 # number of lyapunov steps to run
    rho = 1
    n_pts = 100
    
    # fix L = 1 for now, we'll study its dependency later
    L = 1
    
    for mu, K in itertools.product(args.mu_list, args.K_list):
        gamma_hi, beta_hi = get_gamma_beta_pair(mu, L, K)
        gamma_lo = 0
        beta_lo = np.maximum(0, (gamma_hi*L)/2 - 1)
        
        betas = np.linspace(beta_lo, beta_hi, n_pts+2, endpoint=False)[1:-1]
        gammas = np.linspace(gamma_lo, gamma_hi, n_pts+2, endpoint=False)[1:-1]
        
        a_vs_beta = np.zeros_like(betas)
        b_vs_beta = np.zeros_like(betas)
        c_vs_beta = np.zeros_like(betas)
        d_vs_beta = np.zeros_like(betas)
        e_vs_beta = np.zeros_like(betas)
        f_vs_beta = np.zeros_like(betas)
        p1_vs_beta = np.zeros_like(betas)
        p2_vs_beta = np.zeros_like(betas)
        
        a_vs_gamma = np.zeros_like(gammas)
        b_vs_gamma = np.zeros_like(gammas)
        c_vs_gamma = np.zeros_like(gammas)
        d_vs_gamma = np.zeros_like(gammas)
        e_vs_gamma = np.zeros_like(gammas)
        f_vs_gamma = np.zeros_like(gammas)
        p1_vs_gamma = np.zeros_like(gammas)
        p2_vs_gamma = np.zeros_like(gammas)
        
        for i in range(n_pts):            
            # fix gamma and sweep over beta
            beta = betas[i]
            value, _, P, p, _, _ = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma_hi, mu, L, rho, T, return_all=True)
            if value != 0.:
                # print("Verification failed for")
                # print("K=%.2f, mu=%.2f, gamma=%.2f, beta=%.2f" % (K, mu, gamma_hi, beta))
                
                # ignore this beta as we may be in a region where the lower 
                # bound is above the stability line
                b = -1
                a = -1
                c = -1
                d = -1
                e = -1
                f = -1
                p1 = -1
                p2 = -1
                
                a_vs_beta[i] = a
                b_vs_beta[i] = b
                c_vs_beta[i] = c
                d_vs_beta[i] = d
                e_vs_beta[i] = a
                f_vs_beta[i] = b
                p1_vs_beta[i] = c
                p2_vs_beta[i] = d
            else:
                a = P.value[0,0]
                b = P.value[2,2]
                c = P.value[3,3]
                d = P.value[1,2]
                e = P.value[1,3]
                f = P.value[2,3]
                p1 = p.value[0]
                p2 = p.value[1]
                
                a_vs_beta[i] = a
                b_vs_beta[i] = b
                c_vs_beta[i] = c
                d_vs_beta[i] = d
                e_vs_beta[i] = e
                f_vs_beta[i] = f
                p1_vs_beta[i] = p1
                p2_vs_beta[i] = p2
            
            # fix beta and sweep over ganna
            gamma = gammas[i]
            value, _, P, p, _, _ = hblyap.lyapunov_heavy_ball_momentum_multistep(beta_hi, gamma, mu, L, rho, T, return_all=True)
            assert value != inf and value != -inf

            a = P.value[0,0]
            b = P.value[2,2]
            c = P.value[3,3]
            d = P.value[1,2]
            e = P.value[1,3]
            f = P.value[2,3]
            p1 = p.value[0]
            p2 = p.value[1]
            
            a_vs_gamma[i] = a
            b_vs_gamma[i] = b
            c_vs_gamma[i] = c
            d_vs_gamma[i] = d
            e_vs_gamma[i] = e
            f_vs_gamma[i] = f
            p1_vs_gamma[i] = p1
            p2_vs_gamma[i] = p2
            
        # make a plot for this mu and K
        nrows = 8
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(2.5*ncols,2.5*nrows),
                                constrained_layout=True)
                
        for j, (params, label) in enumerate(zip([a_vs_gamma, b_vs_gamma, c_vs_gamma, d_vs_gamma, e_vs_gamma, f_vs_gamma, p1_vs_gamma, p2_vs_gamma], ["a", "b", "c", "d", "e", "f", "p1", "p2"])):
            ax = axs[j,0]
            ax.scatter(gammas, params, s=1)
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            if j == 0:
                ax.set_title(r"$\bar\beta=%.2f$" % beta_hi, fontsize=17) 
                
            if j == nrows - 1:
                ax.set_xlabel(r"$\gamma$", fontsize=17)
                
        for j, (params, label) in enumerate(zip([a_vs_beta, b_vs_beta, c_vs_beta, d_vs_beta, e_vs_beta, f_vs_beta, p1_vs_beta, p2_vs_beta], ["a", "b", "c", "d", "e", "f", "p1", "p2"])):
            ax = axs[j,1]
            ind = np.where(params < 0)
            if len(ind) == n_pts:
                continue
            elif len(ind) > 0:
                betas_tmp = np.delete(betas, ind)
                params = np.delete(params, ind)
            else:
                betas_tmp = betas
                
            ax.scatter(betas_tmp, params, s=1)
            
            if j == 0:
                ax.set_title(r"$\bar\gamma=%.2f$" % gamma_hi, fontsize=17)
            
            if j == nrows - 1:
                ax.set_xlabel(r"$\beta$", fontsize=17)

        fig.suptitle(r"$K=%.2f, \, \mu=%.2f$" % (K, mu), fontsize=17)
        
        # save
        figname = "lyap_param_vs_gamma_and_beta_K=%.2f_mu=%.2f.png" % (K, mu)
        fig_fn = os.path.join(TMP_DIR, figname)
        plt.savefig(fig_fn)
        # plt.savefig(fig_fn.replace("png", "pdf"))
        print("Figure saved at \n%s" % fig_fn)
        plt.close()

import os
import pickle
import argparse

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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-mu', '--mu_list', nargs='+', type=float)
                            
    # args = parser.parse_args()
    
    # setup
    T = 1 # number of lyapunov steps to run
    rho = 1
    
    # fix L = 1 for now, we'll study its dependency later
    L = 1

    # varying K and mu
    n_pts = 100
    
    logname = "lyap_param_vs_both_K_and_mu.pkl"
    log_fn = os.path.join(LOG_DIR, logname)
    if os.path.exists(log_fn):
        with open(log_fn, "rb") as f:
            res = pickle.load(f)
            
        mus = res["mu"]
        Ks = res["K"]
        a_list = res["a"]
        b_list = res["b"]
        c_list = res["c"]
        d_list = res["d"]
        
    else:
        mu_list = np.linspace(0, 1, n_pts+2, endpoint=False)[1:-1]
        K_list = np.linspace(2, 10, n_pts+1, endpoint=False)[1:]
        mus, Ks = np.meshgrid(mu_list, K_list)
        
        mus = mus.flatten()
        Ks = Ks.flatten()
    
        a_list = np.zeros_like(Ks)
        b_list = np.zeros_like(Ks)
        c_list = np.zeros_like(Ks)
        d_list = np.zeros_like(Ks)

        for i in tqdm(range(len(Ks))):
            mu, K = mus[i], Ks[i]
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
        
        res = {
            "a": a_list,
            "b": b_list,
            "c": c_list,
            "d": d_list,
            "mu" : mus,
            "K" : Ks,
        }
        with open(log_fn, "wb") as f:
            pickle.dump(res, f)

    # make plots
    nrows = 4
    ncols = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(2.5*ncols,2.5*nrows),
                            constrained_layout=True)


    for j, (params, label) in enumerate(zip([a_list, b_list, c_list, d_list], ["a", "b", "c", "d"])):
        ax = axs[j]
        sh = (n_pts, n_pts)
        if label != "d":
            params = np.log(params)
        ax.contourf(mus.reshape(sh), Ks.reshape(sh), params.reshape(sh), 
                    levels=10, cmap=CONTOUR_CMAP)
        ax.set_xlabel(r"$\mu$", fontsize=17)
        ax.set_ylabel(r"$K$", fontsize=17)
        ax.set_title(r"$%s$" % label, fontsize=17)    
        
    # save
    figname = "lyap_param_vs_both_K_and_mu.png"
    fig_fn = os.path.join(TMP_DIR, figname)
    plt.savefig(fig_fn)
    # plt.savefig(fig_fn.replace("png", "pdf"))
    print("Figure saved at \n%s" % fig_fn)

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
    nrows = 15
    ncols = 2
    fig_all, axs_all = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(2.5*ncols,2.5*nrows),
                            constrained_layout=True)
    colors = mpl.cm.viridis(np.linspace(0,1,len(K_list)))
    for i_K, K in enumerate(K_list):
        n_pts = 100
        mu_list = np.linspace(0, 1, n_pts+2, endpoint=False)[1:-1]
        beta_list = np.zeros_like(mu_list)
        gamma_list = np.zeros_like(mu_list)
        
        a_list = np.zeros_like(mu_list)
        b_list = np.zeros_like(mu_list)
        c_list = np.zeros_like(mu_list)
        d_list = np.zeros_like(mu_list)
        e_list = np.zeros_like(mu_list)
        f_list = np.zeros_like(mu_list)
        p1_list = np.zeros_like(mu_list)
        p2_list = np.zeros_like(mu_list)
        l0_list = np.zeros_like(mu_list)
        l1_list = np.zeros_like(mu_list)
        l2_list = np.zeros_like(mu_list)
        l3_list = np.zeros_like(mu_list)
        l4_list = np.zeros_like(mu_list)
        l5_list = np.zeros_like(mu_list)
        Res = [np.zeros_like(mu_list) for _ in range(15)]

        for i, mu in enumerate(mu_list):
            _, beta = get_gamma_beta_pair(mu, L, K)
            kappa = mu/L
            gamma = (1-beta)/((1-beta*kappa)**2)
            gamma *= (1+3*beta*(1-kappa)-kappa*beta**2 + np.sqrt(4*beta*(2-kappa-kappa*beta)*(1+beta-2*beta*kappa)) )
            if T > 1:
                beta = bisection_max_beta(beta, gamma, mu, L, rho, T)
            
            gamma_max = gamma
            gamma_min = 0
            value, diagnostics = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta, gamma, mu, L, rho, T, return_all=True)
            # assert value != inf
            if value == inf:
            # find the largest gamma that works for this beta
                while gamma_max - gamma_min > 1e-10:
                    gamma_test = (gamma_max + gamma_min) / 2
                    value, diagnostics = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta, gamma, mu, L, rho, T, return_all=True)
                    if value == inf:
                        gamma_max = gamma_test
                    else:
                        gamma_min = gamma_test
                        
                gamma = gamma_min
                
            # final solve
            value, diagnostics = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta, gamma, mu, L, rho, T, return_all=True)
            P, p, A_lb, a_lb, dual_n, dual_m, R_vec_m, R_mat_m, R_vec_n, R_mat_n = diagnostics
            # rho real can be derived from R11 - R22 (scaled)
            rho_real = 1 - 2*(L-mu)*(R_mat_m[0,0] - R_mat_m[1,1]) /(-2*mu*L*p[0]*(1+beta)**2)
                
            phi = A_lb.data[0][2]
            theta = A_lb.data[0][0]
                
            a = P[0,0]
            b = P[2,2]
            c = P[1,2]
            d = P[0,3]
            e = gamma
            f = beta
            p1 = p[0]
            # p2 = d/(p[0]*(1+beta-rho))
            # p2 = R_mat_m[4,4]*2*(L-mu) / d
            # p2 = mu*L*p[0]*(2-rho+beta*(2+beta-2*rho))
            # p2 = -2*(L-mu)*a*(beta**2-rho)
            p2 =  b/(1-beta) * (L-mu) #- mu*L*(2*p[0] + mu*L)
                        
            l0 = A_lb.data[0][0]
            l1 = A_lb.data[0][1]
            l2 = A_lb.data[0][2]
            l3 = A_lb.data[0][3]
            l4 = a_lb[0]
            l5 = a_lb[1]
            
            a_list[i] = a
            b_list[i] = b
            c_list[i] = c
            d_list[i] = d
            e_list[i] = e
            f_list[i] = f
            p1_list[i] = p1
            p2_list[i] = p2
            l0_list[i] = l0
            l1_list[i] = l1
            l2_list[i] = l2
            l3_list[i] = l3
            l4_list[i] = l4
            l5_list[i] = l5
            
            beta_list[i] = beta
            gamma_list[i] = gamma

            idx = 0
            for ell in range(5):
                for j in range(5):
                    if ell <= j:
                        Res[idx][i] = R_mat_m[ell, j]*2*(L-mu)
                        idx += 1
            
            # idx = 0
            # for ell in range(4):
            #     for j in range(4):
            #         if ell <= j:
            #             Res[idx][i] = R_mat_n[ell, j]*2*(L-mu)
            #             idx += 1

        # make plots
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(2.5*ncols,2.5*nrows),
                                constrained_layout=True)


        for j, (params, label) in enumerate(zip([a_list, b_list, c_list, d_list, e_list, f_list, p1_list, p2_list, l0_list, l1_list, l2_list, l3_list, l4_list, l5_list], ["a", "b", "c", "d", "e", "f", r"$p_1$", r"$p_2$", "l0", "l1", "l2", "l3", "l4", "l5"])):
            ax = axs[j,0]
            ax.plot(mu_list, params, linewidth=2, color=colors[i_K])
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    

            if j == nrows - 1:
                ax.set_xlabel(r"$\mu$", fontsize=17)
                
            ax = axs_all[j,0]
            lab = r"$K=%.2f$" % K
            ax.plot(mu_list, params, linewidth=2, color=colors[i_K], label=lab)
            ax.set_ylabel(r"$%s$" % label, fontsize=17)    
            
            # if label == r"$p_1$":
                # compare = a_list*gamma
                # ax.plot(mu_list, compare, linewidth=2, color=colors[i_K], linestyle="--", marker=".", markevery=10)

            if j == nrows - 1:
                ax.set_xlabel(r"$\mu$", fontsize=17)
            
            # ax.set_yscale("log")
            
        idx = 0
        for ell in range(5):
            for j in range(5):
                if ell <= j:
                    ax = axs_all[idx,1] 
                    params = Res[idx]
                    ax.plot(mu_list, params, linewidth=2, color=colors[i_K], label=lab)
                    ax.set_ylabel(r"$R_{%d,%d}$" % (ell+1, j+1), fontsize=17)
                    
                    if idx == nrows - 1:
                        ax.set_xlabel(r"$\mu$", fontsize=17)
    
                    idx += 1
        
        # idx = 0
        # for ell in range(4):
        #     for j in range(4):
        #         if ell <= j:
        #             ax = axs_all[idx,1] 
        #             params = Res[idx]
        #             ax.plot(mu_list, params, linewidth=2, color=colors[i_K], label=lab)
        #             ax.set_ylabel(r"$R_{%d,%d}$" % (ell+1, j+1), fontsize=17)
                    
        #             if idx == nrows - 1:
        #                 ax.set_xlabel(r"$\mu$", fontsize=17)
    
        #             idx += 1

        fig.suptitle(r"$K=%.2f$" % K, fontsize=17)

        # save
        figname = "lyap_param_vs_mu_K=%.2f.png" % K
        fig_fn = os.path.join(TMP_DIR, figname)
        fig.savefig(fig_fn)
        # plt.savefig(fig_fn.replace("png", "pdf"))
        print("Figure saved at \n%s" % fig_fn)

    # save
    axs_all[2,0].legend(frameon=False)
    figname = "lyap_param_vs_mu_all.png"
    fig_fn = os.path.join(TMP_DIR, figname)
    fig_all.savefig(fig_fn)
    # plt.savefig(fig_fn.replace("png", "pdf"))
    print("Figure saved at \n%s" % fig_fn)

import argparse
from math import inf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import algorithms.heavy_ball.lyapunov as hblyap


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": "Helvetica",
    "figure.dpi": 600
})

TMP_DIR = "tmp"
LOG_DIR = "log"


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


def gamma_beta_pair_on_boundary(K, kappa):
    phi = np.cos( 2*np.pi / K )
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
    parser.add_argument('-mu', '--mu', type=float)
    parser.add_argument('-K', '--K', type=float)
    parser.add_argument('-T', '--lyapunov_steps', type=int)
                            
    args = parser.parse_args()
    
    L = 1
    mu = args.mu
    kappa = mu/L
    K = args.K
    
    assert K >= 2
    
    gamma, beta = gamma_beta_pair_on_boundary(K, kappa)
    rho = 1
    lyapunov_steps = args.lyapunov_steps
    # a version that guarantees to fill in the gap starting from smooth boundary
    if lyapunov_steps > 1:
        max_beta = bisection_max_beta(beta, gamma, mu, L, rho, lyapunov_steps)
    else:
        max_beta = beta
    print("mu    = ", mu)
    print("gamma = ", gamma)
    print("beta:                   ", beta)
    print("max beta for T=%d steps: " % lyapunov_steps, max_beta)
    
    value, sdp_prob, P, p, dual_n, dual_m = hblyap.lyapunov_heavy_ball_momentum_multistep(max_beta, gamma, mu, L, rho, lyapunov_steps, return_all=True)
    print("\nOptimal value", value, "\n")

    Pmat = P.value
    d = p.value[1]
    b = Pmat[0,3]
    a = Pmat[0,0]
    c = Pmat[3,3]

    print("a       ", a, "        sqrt(a)  ", np.sqrt(a))
    print("b       ", b)
    print("c       ", c, "        sqrt(c)  ", np.sqrt(c))
    print("d       ", d)
    
    print("\n1/beta", 1/beta, "\n")
    print("\nkappa/beta", kappa/beta, "\n")
    
    print("dual variables corresponding to NONNEGATIVITY constraints")
    print(dual_n.value)
    
    mm = lyapunov_steps+3
    M = np.zeros((mm, mm))
    row_idx, col_idx = np.where(~np.eye(lyapunov_steps+3, dtype=bool))
    M[row_idx, col_idx] = dual_m.value
    
    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    ax.imshow(M)
    
    headers = [r"$k-1$", r"$k$"]
    for t in range(1,lyapunov_steps+1):
        headers += [r"$k+%d$"%t]
    headers += [r"*"]
    
    ax.set_xticks(np.arange(len(headers))[::2])
    ax.set_yticks(np.arange(len(headers))[::2])
    ax.set_xticklabels(headers[::2], rotation=45)
    ax.set_yticklabels(headers[::2])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.set_title(r"$T=%d, \, \mu=%.2f$" % (lyapunov_steps, mu), fontsize=17)

    fig_fn = "tmp/dual_m_mu=%.2f_T=%d_K=%.2f.png" % (mu, lyapunov_steps, K)
    plt.savefig(fig_fn)
    print("Figure saved at \n%s" % fig_fn)

    np.set_printoptions(precision=2)
    print("dual variables corresponding to MONOTONICITY constraints")
    print(M)
    

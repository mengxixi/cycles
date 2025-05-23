import argparse
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


max_beta_dict = {
    "0.3" : {
        "1" : 0.5846916894790035,
        "2" : 0.584691705,
        "3" : 0.58563166799,
        "4" : 0.6128960287,
        "5" : 0.621617564,
        "6" : 0.620995589,
        "7" : 0.620878129,
        "8" : 0.623542679,
        "9" : 0.62438227
    },
    "0.7" : {
        "1" : 0.812679644673091,
        "2" : 0.812679658866,
        "3" : 0.81305912,
        "4" : 0.8387521, 
        "5" : 0.8761921999,
        "6" : 0.885219,
        "7" : 0.8866566,   
        "8" : 0.8921705298979,
        "9" : 0.8948179999,        
    }
}


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
    print("mu    = ", mu)
    print("gamma = ", gamma, "beta = ", beta)
    
    lyapunov_steps = args.lyapunov_steps
    beta = max_beta_dict[str(mu)][str(lyapunov_steps)]
    rho = 1

    value, sdp_prob, P, p, dual_n, dual_m = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, lyapunov_steps, return_all=True)
    print("\nOptimal value", value, "\n")

    Pmat = P.value
    d = p.value[1]
    b = Pmat[0,3]
    a = Pmat[0,0]
    c = Pmat[3,3]

    print("a       ", a)
    print("b       ", b)
    print("c       ", c)
    print("d       ", d)

    print("dual variables corresponding to NONNEGATIVITY constraints")
    print(dual_n.value)
    
    print(len(dual_m.value))
    mm = lyapunov_steps+3
    M = np.zeros((mm, mm))
    row_idx, col_idx = np.where(~np.eye(lyapunov_steps+3, dtype=bool))
    M[row_idx, col_idx] = dual_m.value
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(M)
    
    headers = [r"$k-1$", r"$k$"]
    for t in range(1,lyapunov_steps+1):
        headers += [r"$k+%d$"%t]
    headers += [r"*"]
    
    ax.set_xticks(np.arange(len(headers)))
    ax.set_yticks(np.arange(len(headers)))
    ax.set_xticklabels(headers)
    ax.set_yticklabels(headers)
    
    ax.set_title(r"$T=%d, \, \mu=%.2f$" % (lyapunov_steps, mu), fontsize=17)

    fig_fn = "tmp/dual_m_T=%d_mu=%.2f_K=%.2f.png" % (lyapunov_steps, mu, K)
    plt.savefig(fig_fn)
    print("Figure saved at \n%s" % fig_fn)

    np.set_printoptions(precision=4)
    print("dual variables corresponding to MONOTONICITY constraints")
    print(M)
    

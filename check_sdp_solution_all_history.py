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


def intersection_of_cycle_boundary(K1, K2, mu, kappa):
    phi1 = np.cos( 2*np.pi / K1 )
    a = 2*kappa*(1-phi1)
    b = 1
    c = 2*(kappa*phi1-1)
    d = -4*kappa*phi1*(1-phi1)
    e = 2*(phi1-kappa)
    f = 2*kappa*(1-phi1)
    
    phi2 = np.cos( 2*np.pi / K2 )
    ap = 2*kappa*(1-phi2)
    bp = 1
    cp = 2*(kappa*phi2-1)
    dp = -4*kappa*phi2*(1-phi2)
    ep = 2*(phi2-kappa)
    fp = 2*kappa*(1-phi2)
    
    app = a*bp-ap*b
    cpp = c*bp-cp*b
    dpp = d*bp-dp*b
    epp = e*bp-ep*b
    fpp = f*bp-fp*b
    
    def mugamma(x):
        # x = beta
        return -(app*x**2 + dpp*x + fpp) / (cpp*x + epp)

    def func(x):
        # x = beta
        return (a*x**2 + d*x + f)*(cpp*x + epp)**2 \
            + b*(app*x**2 + dpp*x + fpp)**2 \
            - (c*x + e)*(cpp*x + epp)*(app*x**2 + dpp*x + fpp)
    try:
        sol = root_scalar(func, bracket=[0,1], method="brentq")
        beta = sol.root
        gamma = mugamma(beta)/mu
        return (gamma, gamma), beta
    
    except ValueError as err:
        if func(0)*func(1) > 0:
            # safe to assume that we are in the large kappa case where 
            # the two conic curves do not intersect at beta <= 1
            # set beta to 1
            beta = 1
            # compute the gamma that satisfies the original equation 
            # for the two K values
            gamma1 = valid_gamma_for_beta_on_cycle(kappa, mu, phi1, beta, leftroot=True)
            gamma2 = valid_gamma_for_beta_on_cycle(kappa, mu, phi2, beta, leftroot=False)
            
            return (gamma2, gamma1), beta


def valid_gamma_for_beta_on_cycle(kappa, mu, phi, beta, leftroot=True):
    a = 1
    b = -2*(beta-phi+kappa*(1-beta*phi))
    c = 2*kappa*(1-phi)*(1+beta**2-2*beta*phi)
    assert b**2 >= 2*a*c
    
    if leftroot:
        mugamma = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    else:
        mugamma = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return mugamma/mu
    
    
def bisection_max_gamma(gamma_max, beta, mu, L, rho, lyapunov_steps):
    gamma_min = 0
    while gamma_max - gamma_min > 1e-12:
        gamma_next = (gamma_max + gamma_min) / 2
        value = hblyap.lyapunov_heavy_ball_momentum_multistep_all_history(beta, gamma_next, mu, L, rho, lyapunov_steps=lyapunov_steps)
        if value != inf:
            gamma_min = gamma_next
        else:
            gamma_max = gamma_next
    
    value, sdp_prob, P, p, dual_n, dual_m = hblyap.lyapunov_heavy_ball_momentum_multistep_all_history(beta, gamma_min, mu, L, rho, lyapunov_steps=lyapunov_steps, return_all=True)
    
    return gamma_min, value, sdp_prob, P, p, dual_n, dual_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mu', '--mu', type=float)
    parser.add_argument('-K1', '--K1', type=int)
    parser.add_argument('-K2', '--K2', type=int)
    parser.add_argument('-T', '--lyapunov_steps', type=int)
                            
    args = parser.parse_args()
    
    L = 1
    mu = args.mu
    kappa = mu/L
    K1 = args.K1
    K2 = args.K2
        
    (gamma_l, gamma_r), beta = intersection_of_cycle_boundary(K1, K2, mu, kappa)
    gamma_boundary = (gamma_l + gamma_r) / 2
    
    # do a bisection search anyway
    rho = 1
    lyapunov_steps = args.lyapunov_steps
    gamma, value, sdp_prob, P, p, dual_n, dual_m = bisection_max_gamma(gamma_boundary, beta, mu, L, rho, lyapunov_steps)
    print("gamma", gamma, "gamma boundary", gamma_boundary, "beta", beta)

    print("\nOptimal value", value, "\n")
    
    np.set_printoptions(precision=6)
    P = P.value
    p = p.value
    print("P\n")
    print(P)
    
    print("p\n")
    print(p)
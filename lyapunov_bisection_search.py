from tqdm import tqdm
from joblib import Parallel, delayed

from math import inf
import numpy as np

import tools.cycle_utils as cu
from tools.file_management import write_result_file, write_result_file_multistep
from algorithms.heavy_ball.lyapunov import lyapunov_heavy_ball_momentum_multistep, lyapunov_heavy_ball_momentum_multistep_fixed
from algorithms.nag.lyapunov import lyapunov_accelerated_gradient_strongly_convex
from algorithms.inexact_gradient_descent.lyapunov import lyapunov_inexact_gradient_descent
from algorithms.three_operator_splitting.lyapunov import lyapunov_three_operator_splitting


def lyapunov_bisection_search(method, mu, L, nb_points, precision, rho=1):
    if method == "TOS":
        betas = np.linspace(0, 2, nb_points + 1, endpoint=False)[1:]
    else:
        betas = np.linspace(0, 1, nb_points + 1, endpoint=False)[1:]
    gammas_min_lyap = np.zeros_like(betas)
    gammas_max_lyap = [cu.bound(method=method, L=L, beta=beta) for beta in betas]
    if method == "HB":
        lyapunov_search = lyapunov_heavy_ball_momentum_multistep
    elif method == "NAG":
        lyapunov_search = lyapunov_accelerated_gradient_strongly_convex
    elif method == "GD":
        lyapunov_search = lyapunov_inexact_gradient_descent
    elif method == "TOS":
        lyapunov_search = lyapunov_three_operator_splitting
    else:
        raise ValueError
    gammas_lyap = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        gamma_min_lyap = gammas_min_lyap[it]
        gamma_max_lyap = gammas_max_lyap[it]

        while gamma_max_lyap - gamma_min_lyap > precision:
            gamma = (gamma_min_lyap + gamma_max_lyap) / 2
            lyap = lyapunov_search(beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, )
            if lyap != inf:
                gamma_min_lyap = gamma
            else:
                gamma_max_lyap = gamma

        gammas_lyap.append(gamma_min_lyap)

    logdir = "results/lyapunov"
    fn = "{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L)
    write_result_file(logdir=logdir, filename=fn,
                      gammas=gammas_lyap, betas=betas)


def get_HB_gamma_bisection_intervals_for_beta(mu, L, beta, lyapunov_steps):
    b_max = cu.bound("HB", L, beta)
    
    cycle_intervals = cu.get_cycle_intervals_for_beta(mu, L, beta)
    bisect_intervals = cu.complement_of_intervals_within_bounds(cycle_intervals, 0, b_max)

    return bisect_intervals


def lyapunov_bisection_search_multistep(method, mu, L, nb_points, precision, rho=1, lyapunov_steps=1):
    if method != "HB":
        raise NotImplementedError
    lyapunov_search = lyapunov_heavy_ball_momentum_multistep
    # lyapunov_search = lyapunov_heavy_ball_momentum_multistep_fixed

    all_valid_gamma_intervals = []
    
    betas = np.linspace(0, 1, nb_points + 1, endpoint=False)[1:]
    for it in tqdm(range(len(betas))):
        beta = betas[it]
        
        # generate a list of gamma intervals, each to be bisect on    
        intervals = get_HB_gamma_bisection_intervals_for_beta(mu=mu, L=L, beta=beta, lyapunov_steps=lyapunov_steps)

        valid_gamma_intervals = []
        for (gamma_min, gamma_max) in intervals:
            gamma_min_lyap = gamma_min
            gamma_max_lyap = gamma_max
            
            while gamma_max_lyap - gamma_min_lyap > precision:
                gamma = (gamma_min_lyap + gamma_max_lyap) / 2
                lyap = lyapunov_search(beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, lyapunov_steps=lyapunov_steps)
                if lyap != inf:
                    gamma_min_lyap = gamma
                else:
                    gamma_max_lyap = gamma
                    
            valid_gamma_intervals += [ (gamma_min, gamma_min_lyap) ]

        all_valid_gamma_intervals.append(valid_gamma_intervals)
        
    # write results to file
    logdir = "results/lyapunov"
    fn = "{}_mu{:.2f}_L{:.0f}_steps_{:d}.txt".format(method, mu, L, lyapunov_steps)
    write_result_file_multistep(logdir=logdir, filename=fn, 
                                gamma_intervals=all_valid_gamma_intervals, 
                                betas=betas)
    

if __name__ == "__main__":

    Parallel(n_jobs=-1)(delayed(lyapunov_bisection_search)(method=method,
                                                           mu=0.1,
                                                           L=1,
                                                           nb_points=300,
                                                           precision=10 ** -4,
                                                           rho=1,
                                                           ) for method in ["HB"])

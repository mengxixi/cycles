from joblib import Parallel, delayed

import numpy as np

from lyapunov_bisection_search import lyapunov_bisection_search, lyapunov_bisection_search_multistep
from cycle_bisection_search import cycle_bisection_search
from tools.file_management import get_colored_graphics, get_colored_graphics_HB_multistep_lyapunov, get_colored_graphics_HB_lyapunov_all_history


def run_all(list_algos, list_mus, nb_points, precision, max_cycle_length):
    methods = list()
    mus = list()
    for method in list_algos:
        for mu in list_mus:
            methods.append(method)
            mus.append(mu)

    Parallel(n_jobs=-1)(delayed(lyapunov_bisection_search)(method=methods[i],
                                                           mu=mus[i],
                                                           L=1,
                                                           nb_points=nb_points,
                                                           precision=precision,
                                                           rho=1,
                                                           ) for i in range(len(methods)))

    methods = list()
    mus = list()
    cycle_lengths = list()
    for method in list_algos:
        for mu in list_mus:
            for cycle_length in range(2, max_cycle_length + 1):
                methods.append(method)
                mus.append(mu)
                cycle_lengths.append(cycle_length)

    Parallel(n_jobs=-1)(delayed(cycle_bisection_search)(method=methods[i],
                                                        mu=mus[i],
                                                        L=1,
                                                        nb_points=nb_points,
                                                        precision=precision,
                                                        cycle_length=cycle_lengths[i],
                                                        ) for i in range(len(methods)))

    for method in list_algos:
        for mu in list_mus:
            try:
                get_colored_graphics(method=method, mu=mu, L=1, max_cycle_length=max_cycle_length)
            except FileNotFoundError:
                pass


def run_HB_multistep_lyapunov(list_mus, nb_points, precision, max_lyapunov_steps):
    """A runner similar to run_all but only for HB, where the lyapunov search 
    is performed over multiple steps. Cycle search is also removed as we have 
    analytical expressions for the cycle boundary.
    """
    
    list_algos = ["HB"]
    
    methods = list()
    mus = list()
    lyapunov_steps = list()
    for method in list_algos:
        for mu in list_mus:
            for step in range(1, max_lyapunov_steps + 1):
                methods.append(method)
                mus.append(mu)
                lyapunov_steps.append(step)

    Parallel(n_jobs=-1)(delayed(lyapunov_bisection_search_multistep)(method=methods[i],
                                                                        mu=mus[i],
                                                                        L=1,
                                                                        nb_points=nb_points,
                                                                        precision=precision,
                                                                        rho=1,
                                                                        lyapunov_steps=lyapunov_steps[i],
                                                                     ) for i in range(len(methods)))

    for method in list_algos:
        for mu in list_mus:
            # get_colored_graphics_HB_lyapunov_all_history(mu=mu, L=1, max_lyapunov_steps=max_lyapunov_steps)
            get_colored_graphics_HB_multistep_lyapunov(mu=mu, L=1, max_lyapunov_steps=max_lyapunov_steps)


if __name__ == "__main__":

    # run_all(list_algos=["HB"], list_mus=[0.1], nb_points=300, precision=10**-4, max_cycle_length=25)
    
    nb_points = 300
    # list_mus = np.linspace(0.1, 0.9, 9, endpoint=True)
    # list_mus = [0.7]
    list_mus = [0.1, 0.3, 0.7, 0.9]
    run_HB_multistep_lyapunov(list_mus=list_mus, nb_points=nb_points, precision=10**-4, max_lyapunov_steps=10)

import os
import numpy as np
import cvxpy as cp

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction

import matplotlib.pyplot as plt
from tools.interpolation_conditions import smooth_strongly_convex_matrix_i_j

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": "Helvetica",
    "figure.dpi": 600
})

TMP_DIR = "tmp"

def symmetrize(A):
    return (np.triu(A) + np.triu(A).T) - np.diagflat(np.diagonal(A))


def get_A_matrices(mu, L, gamma):
    A1 = symmetrize([
        [0, 0, 0],
        [0, (gamma-1/L)**2 *(mu*L)/(2*(L-mu)) + 1/(2*L), gamma/2 - 1/(2*L) + (gamma-1/L)*mu/(2*(L-mu))],
        [0, 0, 1/(2*(L-mu))]
    ])
    
    A2 = symmetrize([
        [0, 0, 0],
        [0, (gamma-1/L)**2 *(mu*L)/(2*(L-mu)) + 1/(2*L) - gamma, -1/(2*L) + (gamma-1/L)*mu/(2*(L-mu))],
        [0, 0, 1/(2*(L-mu))]
    ])
    
    A3 = symmetrize([
        [mu*L/(2*(L-mu)), -mu/(2*(L-mu)), 0],
        [0, 1/(2*(L-mu)), 0],
        [0, 0, 0]
    ])
    
    A4 = symmetrize([
        [mu*L/(2*(L-mu)), -L/(2*(L-mu)), 0],
        [0, 1/(2*(L-mu)), 0],
        [0, 0, 0]
    ])
    
    A5 = symmetrize([
        [mu*L/(2*(L-mu)), -gamma*mu*L/(2*(L-mu)), -mu/(2*(L-mu))],
        [0, gamma**2 * mu*L/(2*(L-mu)), gamma*mu/(2*(L-mu))],
        [0, 0, 1/(2*(L-mu))]
    ])
    
    A6 = symmetrize([
        [mu*L/(2*(L-mu)), -gamma*mu*L/(2*(L-mu)), -L/(2*(L-mu))],
        [0, gamma**2 * mu*L/(2*(L-mu)), gamma*L/(2*(L-mu))],
        [0, 0, 1/(2*(L-mu))]
    ])
    
    xk, gk, gkk = np.eye(3)
    xs = np.zeros(3)
    gs = np.zeros(3)
    xkk = xk - gamma*gk
    
    pointk = (xk, gk)
    pointkk = (xkk, gkk)
    points = (xs, gs)
    
    A1 = smooth_strongly_convex_matrix_i_j(pointk, pointkk, mu, L)
    A2 = smooth_strongly_convex_matrix_i_j(pointkk, pointk, mu, L)
    A3 = smooth_strongly_convex_matrix_i_j(pointk, points, mu, L)
    A4 = smooth_strongly_convex_matrix_i_j(points, pointk, mu, L)
    A5 = smooth_strongly_convex_matrix_i_j(pointkk, points, mu, L)
    A6 = smooth_strongly_convex_matrix_i_j(points, pointkk, mu, L)

    return [A1, A2, A3, A4, A5, A6]


def solve_dual_SDP(mu, L, gamma):
    # define the constant matrices
    As = get_A_matrices(mu, L, gamma)

    lambdas = cp.Variable(6)
    tau = cp.Variable()
    
    S = 0
    for l, A in zip(lambdas, As):
        S += l*A
    
    constraints = []

    constraints += [ -lambdas[0] + lambdas[1] - lambdas[2] + lambdas[3] + tau == 0 ]
    constraints += [ lambdas[0] - lambdas[1] - lambdas[4] + lambdas[5] == 1 ]
    constraints += [ lambdas[2] - lambdas[3] + lambdas[4] - lambdas[5] - tau == -1 ]
    constraints += [ S >> 0 ]
    
    constraints += [ lambdas >= 0 ]
    constraints += [ tau >= 0 ]
    
    # additional constraints to simplify the dual variables
    # constraints += [ lambdas[1] == 0 ]
    # constraints += [ lambdas[2] == 0 ]
    # constraints += [ lambdas[4] == 0 ]

    sqrttau = np.maximum(np.abs(1-gamma*mu), np.abs(1-gamma*L))
    constraints += [ lambdas[0] == sqrttau ]
    constraints += [ lambdas[5] == 1-sqrttau ]
    constraints += [ lambdas[3] == sqrttau - sqrttau**2]
    
    prob = cp.Problem(cp.Minimize(tau), constraints=constraints)
    min_tau = prob.solve(solver="MOSEK")
    
    return min_tau, [lambdas[i].value for i in range(6)]
    

def solve_primal_SDP(mu, L, gamma):
    As = get_A_matrices(mu, L, gamma)
    
    G = cp.Variable((3,3), PSD=True)
    fvars = cp.Variable(2)
    fk = fvars[0]
    fkk = fvars[1]
    fs = 0
    
    constraints = []
    constraints += [ fkk - fk + cp.trace(As[0]@G) <= 0 ]
    constraints += [ fk - fkk + cp.trace(As[1]@G) <= 0 ]
    constraints += [ fs - fk + cp.trace(As[2]@G) <= 0 ]
    constraints += [ fk - fs + cp.trace(As[3]@G) <= 0 ]
    constraints += [ fs - fkk + cp.trace(As[4]@G) <= 0 ]
    constraints += [ fkk - fs + cp.trace(As[5]@G) <= 0 ]
    constraints += [ fk - fs <= 1 ]
    
    prob = cp.Problem(cp.Maximize(fkk), constraints=constraints)
    max_rhosq = prob.solve(solver="MOSEK")
    
    return max_rhosq


def solve_primal_PEP(mu, L, gamma, return_G=False):
    problem = PEP()
    
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    
    xs = func.stationary_point()
    fs = func(xs)
    
    xk = problem.set_initial_point()
    problem.set_initial_condition(func(xk) - fs <= 1)
    
    xkk = xk - gamma * func.gradient(xk)
    
    problem.set_performance_metric(func(xkk) - fs)
    
    pep_rhosq = problem.solve(verbose=False)
    
    if return_G:
        return pep_rhosq, problem.G_value, problem.F_value
    else:
        return pep_rhosq
    
    
if __name__ == "__main__":
    mu = 0.1
    L = 1
    gammas = np.linspace(-1, 3, num=300)
    
    taus = []
    rhosqs = []
    pep_rhosqs = []
    duals = []
    for gamma in gammas:
        tau, dvals = solve_dual_SDP(mu, L, gamma)
        taus += [tau]
        duals += [dvals]
        
        rhosq = solve_primal_SDP(mu, L, gamma)
        rhosqs += [rhosq]
        
        # pep_rhosq = solve_primal_PEP(mu, L, gamma)
        # pep_rhosqs += [pep_rhosq]
        
    duals = np.array(duals)
    
    nrows = 2
    ncols = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=(2.5*ncols,2.5*nrows),
                           constrained_layout=True)
    ax = axs[0]
    for i in range(6):
        ax.plot(gammas, duals[:,i], label=r"$\lambda_%d$" % (i+1))
    
    ax.set_xlabel(r"$\gamma$", fontsize=17)
    ax.set_ylabel(r"Dual values", fontsize=17)
    ax.set_title(r"$\mu=%.2f$" % mu)
    ax.legend(frameon=False)
    
    ax = axs[1]
    ax.plot(gammas, taus, color="magenta")
    ax.plot(gammas, rhosqs, color="blue", linestyle="--")
    # ax.plot(gammas, pep_rhosqs, color="green", linestyle="--")
    
    # theoretical = np.maximum(np.abs(1-gammas*mu), np.abs(1-gammas*L))**2
    # ax.plot(gammas, theoretical, color="black")
    
    ax.set_xlabel(r"$\gamma$", fontsize=17)
    ax.set_ylabel(r"$\tau$", fontsize=17)
    ax.set_title(r"$\mu=%.2f$" % mu)
    ax.axhline(1.0, linestyle="--", color="grey")
    ax.axvline(2/(L+mu), linestyle="--", color="grey")

    figname = "fval_duals.png"
    fig_fn = os.path.join(TMP_DIR, figname)
    fig.savefig(fig_fn)
    print("Figure saved at \n%s" % fig_fn)
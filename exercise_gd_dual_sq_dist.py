import os
import numpy as np
import cvxpy as cp

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
    Anum = symmetrize([
		[1, -gamma],
		[0, gamma**2]
	])
    
    Adenom = symmetrize([
		[1, 0],
		[0, 0]
	])
    
    xk, gk = np.eye(2)
    xs = np.zeros(2)
    gs = np.zeros(2)
    
    pointk = (xk, gk)
    points = (xs, gs)
    
    A1 = smooth_strongly_convex_matrix_i_j(points, pointk, mu, L)
    A2 = smooth_strongly_convex_matrix_i_j(pointk, points, mu, L)

    return A1, A2, Anum, Adenom


def solve_dual_SDP(mu, L, gamma):
    # define the constant matrices
    A1, A2, Anum, Adenom = get_A_matrices(mu, L, gamma)

    lambdas = cp.Variable(2)
    tau = cp.Variable()
    
    S = 0
    for l, A in zip(lambdas, [A1, A2]):
        S += l*A
    
    constraints = []

    constraints += [ lambdas[0] == lambdas[1] ]
    constraints += [ Anum - tau*Adenom - S << 0 ]
    
    constraints += [ lambdas >= 0 ]
    constraints += [ tau >= 0 ]
    
    prob = cp.Problem(cp.Minimize(tau), constraints=constraints)
    min_tau = prob.solve(solver="MOSEK")
    
    return min_tau, [lambdas[i].value for i in range(2)]
    

def solve_primal_SDP(mu, L, gamma):
    A1, A2, Anum, Adenom = get_A_matrices(mu, L, gamma)
    
    G = cp.Variable((2,2), PSD=True)
    fvars = cp.Variable(2)
    fk = fvars[0]
    fs = fvars[1]
    
    constraints = []
    constraints += [ fs - fk + cp.trace(A1@G) <= 0 ]
    constraints += [ fk - fs + cp.trace(A2@G) <= 0 ]
    constraints += [ cp.trace(Adenom@G) <= 1 ]
    
    prob = cp.Problem(cp.Maximize(cp.trace(Anum@G)), constraints=constraints)
    max_rhosq = prob.solve(solver="MOSEK")
    
    return max_rhosq
    
    
if __name__ == "__main__":
    mu = 0.1
    L = 1
    gammas = np.linspace(-1, 3, num=50)
    
    taus = []
    rhosqs = []
    duals = []
    for gamma in gammas:
        tau, dvals = solve_dual_SDP(mu, L, gamma)
        taus += [tau]
        duals += [dvals]
        
        rhosq = solve_primal_SDP(mu, L, gamma)
        rhosqs += [rhosq]
        
    duals = np.array(duals)
    
    nrows = 2
    ncols = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=(2.5*ncols,2.5*nrows),
                           constrained_layout=True)
    ax = axs[0]
    for i in range(2):
        ax.plot(gammas, duals[:,i], label=r"$\lambda_%d$" % (i+1))
    
    ax.set_xlabel(r"$\gamma$", fontsize=17)
    ax.set_ylabel(r"Dual values", fontsize=17)
    ax.set_title(r"$\mu=%.2f$" % mu)
    ax.legend(frameon=False)
    
    ax = axs[1]
    ax.plot(gammas, taus, color="magenta")
    ax.plot(gammas, rhosqs, color="blue", linestyle="--")
    # ax.plot(gammas, (1-gammas*mu)**2, color="black")
    ax.set_xlabel(r"$\gamma$", fontsize=17)
    ax.set_ylabel(r"$\tau$", fontsize=17)
    ax.set_title(r"$\mu=%.2f$" % mu)
    ax.axhline(1.0, linestyle="--", color="grey")
    ax.axvline(2/(L+mu), linestyle="--", color="grey")

    figname = "sq_dist_duals.png"
    fig_fn = os.path.join(TMP_DIR, figname)
    fig.savefig(fig_fn)
    print("Figure saved at \n%s" % fig_fn)
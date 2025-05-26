import warnings
from math import inf
import numpy as np
import cvxpy as cp

from tools.interpolation_conditions import interpolation_combination


def lyapunov_heavy_ball_momentum(beta, gamma, mu, L, rho):

    # Initialize
    x0, g0, x1, g1, g2 = list(np.eye(5))
    xs = np.zeros(5)
    gs = np.zeros(5)
    f0, f1, f2 = list(np.eye(3))
    fs = np.zeros(3)

    # Run algorithm
    x2 = x1 + beta * (x1 - x0) - gamma * g1

    # Lyapunov
    G = cp.Variable((4, 4), symmetric=True)
    F = cp.Variable((2,))
    list_of_cvxpy_constraints = []

    VG = np.array([x0 - xs, g0, x1 - xs, g1]).T @ G @ np.array([x0 - xs, g0, x1 - xs, g1])
    VG_plus = np.array([x1 - xs, g1, x2 - xs, g2]).T @ G @ np.array([x1 - xs, g1, x2 - xs, g2])
    VF = np.array([f0 - fs, f1 - fs]).T @ F
    VF_plus = np.array([f1 - fs, f2 - fs]).T @ F

    # Write problem
    list_of_points = [(xs, gs, fs), (x0, g0, f0), (x1, g1, f1), (x2, g2, f2)]

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    list_of_cvxpy_constraints.append(VG_plus - rho * VG << matrix_combination)
    list_of_cvxpy_constraints.append(VF_plus - rho * VF <= vector_combination)
    list_of_cvxpy_constraints.append(dual >= 0)

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    list_of_cvxpy_constraints.append(- VG_plus << matrix_combination)
    list_of_cvxpy_constraints.append(f2 - fs - VF_plus <= vector_combination)
    list_of_cvxpy_constraints.append(dual >= 0)

    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=list_of_cvxpy_constraints)
    try:
        value = prob.solve(solver="MOSEK")
    except cp.error.SolverError:
        value = prob.solve(solver="SCS")
    return value


def initialize_lyapunov_heavy_ball_momentum_multistep(K, T):
    """initialize the basis vectors for x, g, and f

    Args:
        K : N or N + 1 where N is the degree of the iterative algorithm
        T : number of Lyapunov steps considered

    """
    N = 1
    dxg = N + K + 2 + (T - 1)
    df = K + T
    
    I = np.eye(dxg)
    i = N + 1
    x_list = list(I[:i, :])
    g_list = list(I[i:, :])
    f_list = list(np.eye(df))
    
    xs = np.zeros(dxg)
    gs = np.zeros(dxg)
    fs = np.zeros(df)
    
    return x_list, g_list, f_list, xs, gs, fs


def get_nonnegativity_constraints(P, p, mu, L):
    K = 1
    
    # Initialize
    x_list, g_list, f_list, xs, gs, fs = initialize_lyapunov_heavy_ball_momentum_multistep(K, T=1)
    
    Vp = p.T @ np.vstack(f_list)
    
    x = np.vstack(x_list)
    g = np.vstack(g_list)
    VP = np.vstack((x, g)).T @ P @ np.vstack((x, g))
    
    # Build constraints
    list_of_points = list(zip(x_list, g_list, f_list))
    list_of_points += [(xs, gs, fs)]
    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    
    constraints = []
    constraints += [ - VP << matrix_combination ]
    # constraints += [ f_list[1] - fs - Vp <= vector_combination ] # break homogeneity
    constraints += [ - Vp <= vector_combination ]
    constraints += [ cp.trace(P) == 1 ] # break homogeneity
    constraints += [ dual >= 0 ]
    
    return constraints, dual


def get_monotonicity_constraints(P, p, beta, gamma, mu, L, rho, lyapunov_steps=1):
    K = 2
    
    # Initialize
    x_list, g_list, f_list, xs, gs, fs = initialize_lyapunov_heavy_ball_momentum_multistep(K, T=lyapunov_steps)

    # Run algorithm
    for t in range(1, lyapunov_steps + 1):
        xprev = x_list[t-1]
        xcurr = x_list[t]
        gcurr = g_list[t]
        
        xnext = xcurr - gamma * gcurr + beta * (xcurr - xprev)
        x_list += [xnext]


    Vp = p.T @ np.vstack(f_list[:2])
    Vp_plus = p.T @ np.vstack(f_list[-2:])
    
    x = np.vstack(x_list[:2])
    g = np.vstack(g_list[:2])
    VP = np.vstack((x, g)).T @ P @ np.vstack((x, g))
    
    x = np.vstack(x_list[-2:])
    g = np.vstack(g_list[-2:])
    VP_plus = np.vstack((x, g)).T @ P @ np.vstack((x, g))

    # Build constraints
    list_of_points = list(zip(x_list, g_list, f_list))
    list_of_points += [(xs, gs, fs)]
    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")

    constraints = []
    constraints += [ VP_plus - rho * VP << matrix_combination ]
    constraints += [ Vp_plus - rho * Vp <= vector_combination ]
    constraints += [ dual >= 0 ]

    return constraints, dual


def lyapunov_heavy_ball_momentum_multistep_fixed(beta, gamma, mu, L, rho, lyapunov_steps=1, return_all=False):   
    c = beta**2 / gamma - mu*beta/2
    b = (2 - gamma*L) / (2*gamma)
    # a2 = beta
    a1 = 1-beta
    
    if np.isclose(b, 0):
        b = 1e-6
        
    p1 = np.maximum(beta, c/b-a1)

    p = np.array([p1, 1])
    P = np.array([
        [b, -b, 0, 0],
        [-b, b, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
                
    # Get constraints
    constraints_n, dual_n = get_nonnegativity_constraints(P, p, mu=mu, L=L)
    constraints_m, dual_m = get_monotonicity_constraints(P, p, beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, lyapunov_steps=lyapunov_steps)
    constraints = constraints_n + constraints_m
    
    # 0 if there exists dual variables such that this P and p combination 
    # is feasible and thus a valid lyapunov function for this gamma, beta combo
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=constraints)
    try:
        value = prob.solve(solver="MOSEK", 
                           verbose=False, 
                           accept_unknown=False,
                        #    mosek_params={}
                           )

    except cp.error.SolverError as e:
        print(e)
        print("Marking problem as infeasible...")
        value = inf
        
    if return_all:
        return value, prob, P, p, dual_n, dual_m
    else:
        return value
    


def lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta, gamma, mu, L, rho, lyapunov_steps=1, return_all=False):
    # Define SDP variables
    P = cp.Variable((4, 4), symmetric=True)
    p = cp.Variable((2,))
    
    # Get constraints
    constraints_n, dual_n = get_nonnegativity_constraints(P, p, mu=mu, L=L)
    constraints_m, dual_m = get_monotonicity_constraints(P, p, beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, lyapunov_steps=lyapunov_steps)
    constraints = constraints_n + constraints_m
    
    # constraints that give us the smooth boundary
    constraints += [ P[2,:] == 0 ] # g_{k-1}
    constraints += [ P[:,2] == 0 ]
    constraints += [ p[0] == 0]

    constraints += [ P[0,0] == P[1,1] ]
    constraints += [ P[0,0] == -P[0,1] ]
    constraints += [ P[0,3] == -P[1,3] ]
    
    # constraints += [ P[0,0] >= 0]
    # constraints += [ P[0,3] >= 0]
    # constraints += [ P[3,3] >= 0]
    # constraints += [ p[1] >= 0]
        
    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=constraints)
    try:
        value = prob.solve(solver="MOSEK", 
                        #    verbose=True, 
                        #    accept_unknown=False,
                        # #    mosek_params={}
                           )

    except cp.error.SolverError as e:
        print(e)
        print("Marking problem as infeasible...")
        value = inf 

    if return_all:
        return value, prob, P, p, dual_n, dual_m
    else:
        return value


def lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, lyapunov_steps=1, return_all=False):
    # Define SDP variables
    P = cp.Variable((4, 4), symmetric=True)
    p = cp.Variable((2,))
    
    # Get constraints
    constraints_n, dual_n = get_nonnegativity_constraints(P, p, mu=mu, L=L)
    constraints_m, dual_m = get_monotonicity_constraints(P, p, beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, lyapunov_steps=lyapunov_steps)
    constraints = constraints_n + constraints_m
    
    # # constraints correspond to Ghadimi's Lyapunov function
    # constraints += [ P[2,:] == 0 ]
    # constraints += [ P[:,2] == 0 ]
    # constraints += [ P[3,:] == 0 ]
    # constraints += [ P[:,3] == 0 ]
    
    # constraints that give us the smooth boundary
    constraints += [ P[2,:] == 0 ] # g_{k-1}
    constraints += [ P[:,2] == 0 ]
    constraints += [ p[0] == 0]

    constraints += [ P[0,0] == P[1,1] ]
    constraints += [ P[0,0] == -P[0,1] ]
    constraints += [ P[0,3] == -P[1,3] ]
    
    # constraints += [ P[0,0] >= 0]
    # constraints += [ P[0,3] >= 0]
    # constraints += [ P[3,3] >= 0]
    # constraints += [ p[1] >= 0]
    
    # # (k,k+1), (k+1,k+2),...
    # ind_nonzero_1 = [(4+lyapunov_steps-1)*(t+1) for t in range(lyapunov_steps)]
    # # (k,k+3), (k+1,k+4),...
    # ind_nonzero_4 = [i+3 for i in ind_nonzero_1[:-3]]
    # ind_nonzero_5 = [i+1 for i in ind_nonzero_4[:-1]]
    # # (k+3,k), (k+4,k+1),...
    # ind_nonzero_3r = [i-3 for i in ind_nonzero_1[3:]]
    # ind_nonzero_3r += [ind_nonzero_3r[-1]+3+lyapunov_steps]
    # ind_nonzero_4r = [i-1 for i in ind_nonzero_3r[1:]]
    # ind_nonzero = ind_nonzero_1 + ind_nonzero_4 + ind_nonzero_5 + ind_nonzero_3r + ind_nonzero_4r
    # ind_m = np.arange(dual_m.shape[0])
    # ind_zero = np.delete(ind_m, ind_nonzero)
    
    # ind_zero = []
    # ind = 0
    # for i in range(lyapunov_steps+3):
    #     for j in range(lyapunov_steps+3):
    #         if i == j:
    #             continue
    #         # all indices coresponding to either (_,*) or (*,_)
    #         if i == lyapunov_steps+2 or j == lyapunov_steps+2:
    #             ind_zero += [ind]
            
    #         # elif j != i+1 and j != i+4 and j != i+5 and i != j+3 and i != j+2:
    #         #     ind_zero += [ind]
            
    #         ind += 1
    # constraints += [ dual_m[ ind_zero ] == 0]
        
    # # the following constraints will kill the multi-step lyapunov verification
    # # constraints += [ dual_m[4] == p[1]/(L-mu)] # vector monotonicity tightness

                        
    # # constraints that preserve the largest green region    
    # constraints += [ P[0,2] == 0 ] # x_{k-1}, g_{k-1}
    # constraints += [ P[2,0] == 0 ]
    
    # constraints += [ P[1,2] == 0 ] # x_k, g_{k-1}
    # constraints += [ P[2,1] == 0 ]
    
    # constraints += [ P[1,3] == 0 ] # x_k, g_k
    # constraints += [ P[3,1] == 0 ]
    
    # constraints += [ P[0,3] == 0 ] # x_{k-1}, g_k
    # constraints += [ P[3,0] == 0 ]
        
    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=constraints)
    try:
        value = prob.solve(solver="MOSEK", 
                        #    verbose=True, 
                        #    accept_unknown=False,
                        # #    mosek_params={}
                           )

    except cp.error.SolverError as e:
        print(e)
        print("Marking problem as infeasible...")
        value = inf 
        # print("try solving with SCS...")
        # value = prob.solve(solver="SCS")
        
        # if value > 1e-3:
        #     warnings.warn("SCS returning suboptimal result, discarding...")
        #     value = inf
        #     P.value = None
        #     p.value = None
        #     dual_m.value = None
        #     dual_n.value = None


    if return_all:
        return value, prob, P, p, dual_n, dual_m
    else:
        return value

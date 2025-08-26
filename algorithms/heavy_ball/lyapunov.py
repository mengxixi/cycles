import warnings
from math import inf
import numpy as np
import cvxpy as cp

from tools.interpolation_conditions import interpolation_combination


def inner_product(u, v):
    matrix = u.reshape(-1, 1) * v.reshape(1, -1)
    return (matrix + matrix.T) / 2


def square(u):
    return inner_product(u, u)


def interpolation_single(pi, pj, mu, L):
    xi, gi, fi = pi
    xj, gj, fj = pj
    
    G = inner_product(gj, xi - xj) + 1 / (2 * L) * square(gi - gj) + mu / (2 * (1 - mu / L)) * square(
        xi - xj - 1 / L * gi + 1 / L * gj)
        
    M = np.array([
        [-mu*L, mu*L, mu, -L],
        [mu*L, -mu*L, -mu, L],
        [mu, -mu, -1, 1],
        [-L, L, 1, -1]])
    
    x = np.vstack([xi, xj])
    g = np.vstack([gi, gj])

    G_ = -0.5 * np.vstack((x, g)).T @ M @ np.vstack((x, g))
    F_ = (fj - fi)*(L-mu)
    
    dual = cp.Variable((1,))
    matrix_combination = dual*G_
    vector_combination = dual*F_
        
    return matrix_combination, vector_combination, dual


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
    list_of_points = [(x0, g0, f0), (x1, g1, f1), (x2, g2, f2), (xs, gs, fs)]

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
        K : N or N + T where N is the degree of the iterative algorithm
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


def get_nonnegativity_constraints_all_history(P, p, beta, gamma, mu, L, lyapunov_steps=1):
    K = 1
    
    # Initialize
    x_list, g_list, f_list, xs, gs, fs = initialize_lyapunov_heavy_ball_momentum_multistep(K, T=lyapunov_steps)
    Vp = p.T @ np.vstack(f_list)

    # Run algorithm
    for t in range(1, lyapunov_steps):
        xprev = x_list[t-1]
        xcurr = x_list[t]
        gcurr = g_list[t]
        
        xnext = xcurr - gamma * gcurr + beta * (xcurr - xprev)
        x_list += [xnext]
    
    x = np.vstack(x_list)
    g = np.vstack(g_list)
    VP = np.vstack((x, g)).T @ P @ np.vstack((x, g))

    # Build constraints
    list_of_points = list(zip(x_list, g_list, f_list))
    list_of_points += [(xs, gs, fs)]
    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    
    constraints = []
    constraints += [ - VP << matrix_combination ]
    constraints += [ f_list[0] - fs - Vp <= vector_combination ] # break homogeneity
    constraints += [ - Vp <= vector_combination ]
    constraints += [ dual >= 0 ]
    
    return constraints, dual


def get_monotonicity_constraints_all_history(P, p, beta, gamma, mu, L, rho, lyapunov_steps=1):
    K = 2
    
    # Initialize
    x_list, g_list, f_list, xs, gs, fs = initialize_lyapunov_heavy_ball_momentum_multistep(K, T=lyapunov_steps)

    # Run algorithm
    for t in range(1, lyapunov_steps+1):
        xprev = x_list[t-1]
        xcurr = x_list[t]
        gcurr = g_list[t]
        
        xnext = xcurr - gamma * gcurr + beta * (xcurr - xprev)
        x_list += [xnext]

    Vp = p.T @ np.vstack(f_list[:-1])
    Vp_plus = p.T @ np.vstack(f_list[1:])

    x = np.vstack(x_list[:-1])
    g = np.vstack(g_list[:-1])
    VP = np.vstack((x, g)).T @ P @ np.vstack((x, g))

    x = np.vstack(x_list[1:])
    g = np.vstack(g_list[1:])
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
    # constraints += [ f_list[0] - fs - Vp <= vector_combination ] # break homogeneity
    constraints += [ - Vp <= vector_combination ]
    # constraints += [ cp.trace(P) + cp.sum(p) == 1 ] # break homogeneity
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
    
    VP_L = VP_plus - rho * VP
    VP_R = matrix_combination
    Vp_L = Vp_plus - rho * Vp
    Vp_R = vector_combination
    
    return constraints, dual, VP_L, VP_R, Vp_L, Vp_R


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
    # constraints_n, dual_n = get_nonnegativity_constraints(P, p, mu=mu, L=L)
    constraints_m, dual_m, VP_L_m, VP_R_m, Vp_L_m, Vp_R_m = get_monotonicity_constraints(P, p, beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, lyapunov_steps=lyapunov_steps)
    constraints = constraints_n + constraints_m

    constraints += [ cp.trace(P) + cp.sum(p) == 1 ] # break homogeneity

    # # Additional constraints
    # constraints += [ P[0,0] == P[1,1] ]
    # constraints += [ P[0,0] == -P[0,1] ]
    # constraints += [ P[0,2] == -P[1,2] ]
    # constraints += [ P[0,3] == -P[1,3] ]
    # constraints += [ P[2,2] == P[3,3] ]
    
    # constraints += [ P[2,2] == (1-beta)/(2*(L-mu)) ]
    
    # constraints += [ p[1] == 1 ]
    # constraints += [ p[0] == 0 ]
    
    # ind_dual_m = [1, 2, 3, 5, 6, 7, 8, 10, 11]
    # constraints += [ dual_m[ind_dual_m] == 0]
    # constraints += [ dual_m[0] == dual_m[9] ]
    # constraints += [ dual_m[4] == 1/(L-mu) ]
    
    # ind_dual_n = [0, 1, 2, 4, 5]
    # constraints += [ dual_n[ind_dual_n] == 0 ]
    # constraints += [ dual_n[3] == 1/(2*(L-mu)) ]
        
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
        return value, prob, P, p, dual_n, dual_m, VP_L_m, VP_R_m, Vp_L_m, Vp_R_m
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
    
    constraints += [ P[0,0] ==  P[1,1] ]
    constraints += [ P[0,0] == -P[0,1] ]
    constraints += [ P[0,2] == -P[1,2] ]
    constraints += [ P[0,3] == -P[1,3] ]
    
    # constraints += [ P[1,2] >= 0 ]
    # constraints += [ P[1,3] >= 0 ]
    # constraints += [ P[2,3] >= 0 ]
    # constraints += [ P[2,2] >= 0 ]
    # constraints += [ P[3,3] >= 0 ]
    
    # constraints += [ p[0] >= 0 ]
    # constraints += [ p[1] >= 0 ]
            
    # try logdet objective
    N = P.shape[0]
    Dk = np.eye(2*N)
    
    log_det_iterations = 1
    log_det_delta = 10 #0000
    for _ in range(log_det_iterations):
        Y = cp.Variable((N,N), symmetric=True)
        Z = cp.Variable((N,N), symmetric=True)
        D = cp.bmat([
            [Y, np.zeros((N, N))],
            [np.zeros((N, N)), Z],
        ])
        
        Inv = np.linalg.inv(Dk + log_det_delta*np.eye(2*N))
        obj_logdet = cp.Minimize(cp.trace(Inv@D))
        
        M = cp.bmat([
            [Y, P],
            [P.T, Z]
        ])
        
        constraints_new = constraints.copy()
        constraints_new += [ M >> 0 ]
        
        prob = cp.Problem(objective=obj_logdet, constraints=constraints_new)
        try:
            value = prob.solve(solver="MOSEK")
            if value < inf:
                Dk = np.bmat([
                    [Y.value, np.zeros((N, N))],
                    [np.zeros((N, N)), Z.value],
                ])
        except cp.error.SolverError as e:
            print(e)
            # break out and ignore the heuristic
            obj = cp.Minimize(0)
            prob = cp.Problem(objective=obj, constraints=constraints)
            try:
                value = prob.solve(solver="MOSEK")
                break
            except cp.error.SolverError as e:
                print(e)
                print("Marking problem as infeasible...")
                value = inf
                break
    
    # # 0 if there exists a Lyapunov
    # # inf otherwise
    # prob = cp.Problem(objective=cp.Minimize(0), constraints=constraints)
    # try:
    #     value = prob.solve(solver="MOSEK", 
    #                     #    verbose=True, 
    #                     #    accept_unknown=False,
    #                     # #    mosek_params={}
    #                        )

    # except cp.error.SolverError as e:
    #     print(e)
    #     print("Marking problem as infeasible...")
    #     value = inf 

    if return_all:
        return value, prob, P, p, dual_n, dual_m
    else:
        return value


def lyapunov_heavy_ball_momentum_multistep_all_history(beta, gamma, mu, L, rho, lyapunov_steps=1, return_all=False):
    # Define SDP variables
    T = lyapunov_steps
    P = cp.Variable((2*(T+1), 2*(T+1)), symmetric=True)
    p = cp.Variable((T+1,))
    
    # Get constraints
    constraints_n, dual_n = get_nonnegativity_constraints_all_history(P, p, beta=beta, gamma=gamma, mu=mu, L=L, lyapunov_steps=lyapunov_steps)
    constraints_m, dual_m = get_monotonicity_constraints_all_history(P, p, beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, lyapunov_steps=lyapunov_steps)
    constraints = constraints_n + constraints_m
    
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

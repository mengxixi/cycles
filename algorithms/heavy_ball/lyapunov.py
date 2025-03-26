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
    # constraints += [ ftt - fs - Vp_plus <= vector_combination ] # no longer using this to break homogeneity
    constraints += [ - Vp <= vector_combination ]
    constraints += [ cp.trace(P) >= 1 ] # use this instead
    constraints += [ dual >= 0 ]
    
    return constraints


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
        
    return constraints


def lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, lyapunov_steps=1, return_all=False):
    # Define SDP variables
    P = cp.Variable((4, 4), PSD=True)
    p = cp.Variable((2,))
    
    # Get constraints
    constraints = get_nonnegativity_constraints(P, p, mu=mu, L=L)
    constraints += get_monotonicity_constraints(P, p, beta=beta, gamma=gamma, mu=mu, L=L, rho=rho, lyapunov_steps=lyapunov_steps)

    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=constraints)
    try:
        value = prob.solve(solver="MOSEK", 
                           verbose=False, 
                           accept_unknown=False,
                        #    mosek_params={
                        #        "MSK_DPAR_INTPNT_QO_TOL_DFEAS" : 1e-10,
                        #        "MSK_DPAR_INTPNT_CO_TOL_PFEAS" : 1e-10,
                        #        "MSK_DPAR_BASIS_TOL_S" : 1e-8,
                        #        }
                           )
    except cp.error.SolverError as e:
        print(e)
        print("try solving with SCS...")
        value = prob.solve(solver="SCS", eps=1e-6, verbose=True)

    if return_all:
        return prob, P, p
    else:
        return value

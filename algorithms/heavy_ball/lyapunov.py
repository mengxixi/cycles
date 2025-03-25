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


def lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, lyapunov_steps=1):

    # TODO: initialize two sets of bases for K=1 and K=2
    K = 2
    
    # Initialize
    x_list, g_list, f_list, xs, gs, fs = initialize_lyapunov_heavy_ball_momentum_multistep(K, lyapunov_steps)
    
    # Run algorithm
    for t in range(lyapunov_steps + 1):
        if t == 0:
            continue
        xprev = x_list[t-1]
        xcurr = x_list[t]
        gcurr = g_list[t]
        
        xnext = xcurr - gamma * gcurr + beta * (xcurr - xprev)
        x_list += [xnext]

    # Lyapunov
    P = cp.Variable((4, 4), symmetric=True)
    p = cp.Variable((2,))
    constraints = []
    
    x0 = x_list[0]; x1 = x_list[1]
    g0 = g_list[0]; g1 = g_list[1]
    f0 = f_list[0]; f1 = f_list[1]
    xt = x_list[-2]; xtt = x_list[-1]
    gt = g_list[-2]; gtt = g_list[-1]
    ft = f_list[-2]; ftt = f_list[-1]
    

    VP = np.array([x0 - xs, g0, x1 - xs, g1]).T @ P @ np.array([x0 - xs, g0, x1 - xs, g1])
    VP_plus = np.array([xt - xs, gt, xtt - xs, gtt]).T @ P @ np.array([xt - xs, gt, xtt - xs, gtt])
    Vp = np.array([f0 - fs, f1 - fs]).T @ p
    Vp_plus = np.array([ft - fs, ftt - fs]).T @ p

    # Write problem
    list_of_points = list(zip(x_list, g_list, f_list))

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    constraints.append(VP_plus - rho * VP << matrix_combination)
    constraints.append(Vp_plus - rho * Vp <= vector_combination)
    constraints.append(dual >= 0)

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    constraints.append(- VP_plus << matrix_combination)
    # constraints.append(ftt - fs - Vp_plus <= vector_combination)
    constraints.append(- Vp_plus <= vector_combination)
    constraints.append(cp.trace(P) >= 1)
    constraints.append(dual >= 0)

    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=constraints)
    try:
        value = prob.solve(solver="MOSEK", verbose=True, 
                           accept_unknown=False,
                        #    mosek_params={
                        #        "MSK_DPAR_INTPNT_QO_TOL_DFEAS" : 1e-10,
                        #        "MSK_DPAR_INTPNT_CO_TOL_PFEAS" : 1e-10,
                        #        "MSK_DPAR_BASIS_TOL_S" : 1e-8,
                        #        }
                           )
    except cp.error.SolverError as e:
        print(e)
        value = prob.solve(solver="SCS", eps=1e-8)
    return value

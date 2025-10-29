import warnings
from math import inf
import numpy as np
import cvxpy as cp


def initialize_points(N, K):
    I = np.eye(N+K+2)
    
    x_list = I[:N+1, :]
    g_list = I[N+1:, :]
    f_list = np.eye(K+1)
    
    xs = np.zeros(N+K+2)
    gs = np.zeros(N+K+2)
    fs = np.zeros(K+1)
    
    return list(x_list), list(g_list), list(f_list), xs, gs, fs


def run_algorithm(x_list_init, g_list, alpha, gammas, betas, N, K):
    x_list = x_list_init.copy()
    y_list = []
    
    for k in range(K+1):
        y_list += [ np.sum(gammas[::-1][:,None] * x_list[k:k+N+1], axis=0) ]
        x_list += [ np.sum(betas[::-1][:,None] * x_list[k:k+N+1], axis=0) - alpha * g_list[k] ]

    return x_list[N:-1], y_list

 
def smooth_strongly_convex_interpolation_i_j(pointi, pointj, mu, L):
    xi, gi, fi = pointi
    xj, gj, fj = pointj
    
    M = np.array([
        [-mu*L, mu*L, mu, -L],
        [mu*L, -mu*L, -mu, L],
        [mu, -mu, -1, 1],
        [-L, L, 1, -1]])
    
    x = np.vstack([xi, xj])
    g = np.vstack([gi, gj])
    
    G = 0.5 * np.vstack((x, g)).T @ M @ np.vstack((x, g))
    F = (fi - fj)*(L-mu)

    return F, G


def get_interpolation_vecs_and_mats(mu, L, x_list, g_list, f_list, xs, gs, fs):
    list_of_points = list(zip(x_list, g_list, f_list))
    list_of_points += [(xs, gs, fs)]
    
    vecs = []
    mats = []
    
    l = len(list_of_points)
    for i in range(l):
        for j in range(l):
            if i != j:
                vec, mat = smooth_strongly_convex_interpolation_i_j(list_of_points[i], list_of_points[j], mu, L)
                vecs += [ vec ]
                mats += [ mat ]

    duals = cp.Variable(len(vecs))
    
    return vecs, mats, duals
    

def get_interpolation_combinations(vecs, mats, duals):
    vec_combinations = 0
    mat_combinations = 0
    
    for i in range(len(vecs)):
        vec_combinations += duals[i] * vecs[i]
        mat_combinations += duals[i] * mats[i]
        
    return vec_combinations, mat_combinations


def lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta, gamma, mu, L, rho, constrain_P=True, lyapunov_steps=1, return_all=False):
    N = 1
    T = lyapunov_steps
    gammas = np.array([1, 0])
    betas = np.array([1+beta, -beta])
    alpha = gamma
    
    # Define SDP variables
    P = cp.Variable((2*(N+1), 2*(N+1)), symmetric=True)
    p = cp.Variable(N+1)
    
    constraints = []
    
    # ========== Define monotonicity constraints ==========
    
    # Initialize
    K = N+T
    x_list_init, g_list, f_list, xs, gs, fs = initialize_points(N, K)
    
    # Append the additional x's resulting from running the algorithm
    x_list, y_list = run_algorithm(x_list_init, g_list, alpha, gammas, betas, N, K)
    
    # Interpolate using y, g, and f (the algorithm assumes that the gradients)
    # are evaluated at the y's, not the x's
    ivecs_m, imats_m, duals_m = get_interpolation_vecs_and_mats(mu, L, y_list, g_list, f_list, xs, gs, fs)
    vec_combinations_m, mat_combinations_m = get_interpolation_combinations(ivecs_m, imats_m, duals_m)
    
    # Build constraints
    fk = np.vstack(f_list[:N+1])
    Sk = np.vstack((np.vstack(x_list[:N+1]), np.vstack(g_list[:N+1])))
    vk_m = p.T@fk
    Vk_m = Sk.T@P@Sk
    
    fkk = np.vstack(f_list[T:N+T+1])
    Skk = np.vstack((np.vstack(x_list[T:N+T+1]), np.vstack(g_list[T:N+T+1])))
    vkk_m = p.T@fkk
    Vkk_m = Skk.T@P@Skk
    
    constraints += [ vkk_m - rho*vk_m + vec_combinations_m <= 0 ]
    constraints += [ Vkk_m - rho*Vk_m + mat_combinations_m << 0 ]
    constraints += [ duals_m >= 0 ]
 
    # ========== Define nonnegativity constraints ==========
    
    # Initialize
    K = N
    x_list_init, g_list, f_list, xs, gs, fs = initialize_points(N, K)
    
    # Append the additional x's resulting from running the algorithm
    x_list, y_list = run_algorithm(x_list_init, g_list, alpha, gammas, betas, N, K)
    
    # Interpolate using y, g, and f (the algorithm assumes that the gradients)
    # are evaluated at the y's, not the x's
    ivecs_n, imats_n, duals_n = get_interpolation_vecs_and_mats(mu, L, y_list, g_list, f_list, xs, gs, fs)
    vec_combinations_n, mat_combinations_n = get_interpolation_combinations(ivecs_n, imats_n, duals_n)
    
    # Build constraints
    fk = np.vstack(f_list[:N+1])
    Sk = np.vstack((np.vstack(x_list[:N+1]), np.vstack(g_list[:N+1])))
    vk_n = p.T@fk
    Vk_n = Sk.T@P@Sk
    
    # this version uses an explicit lower bound
    A = cp.Variable((2*(N+1), 2*(N+1)), diag=True)
    a = cp.Variable(N+1)
    constraints += [ vk_n - a.T@fk - vec_combinations_n == 0 ]
    constraints += [ Vk_n - Sk.T@A@Sk - mat_combinations_n >> 0 ]
    constraints += [ duals_n >= 0 ]
    constraints += [ A >= 0 ]
    constraints += [ a >= 0 ]
    constraints += [ cp.sum(A) + cp.sum(a) >= 1 ]
        
    if constrain_P:
        constraints += [ p[0] == p[1] ]
        constraints += [ p[0] >= 0 ]
        
        constraints += [ P[0,0] == P[1,1] ]
        constraints += [ P[0,0] == -P[0,1] ]
        constraints += [ P[2,2] == P[3,3] ]
        constraints += [ P[2,3] == P[2,2] ]
        
        # constraints += [ P[3,3] == (1-beta)/(2*(L-mu))]
        
        constraints += [ P[0,2] == -P[1,2] ]
        constraints += [ P[0,3] == -P[1,3] ]
        constraints += [ P[0,0] >= 0 ]
        
        ind_dual_m = [1, 2, 3, 5, 6, 7, 8, 9]
        constraints += [ duals_m[ind_dual_m] == 0]
        constraints += [ duals_m[0] == duals_m[4] ]
        constraints += [ duals_m[10] == duals_m[11] ]
        constraints += [ duals_m[0] == p[0]*rho / (L-mu) ]
        constraints += [ duals_m[10] == p[0]*(1-rho) / (L-mu) ]
        
        constraints += [ duals_n[0] == duals_n[2] ]
        constraints += [ duals_n[1] == duals_n[3] ]
        constraints += [ duals_n[4] == duals_n[5] ]
        constraints += [ duals_n[0] == mu*L / (L-mu) ]
        constraints += [ duals_n[1] == p[0] / (L-mu)]
        constraints += [ duals_n[4] == mu*L / (2*(L-mu))]

        constraints += [ A[0,0] == A[1,1] ]
        constraints += [ A[2,2] == A[3,3] ]
        constraints += [ P[0,3] == 1 ]
        
        # these two are implied        
        # constraints += [ a[0] == a[1] ]
        # constraints += [ a[0] == mu*L/2]
 
    prob = cp.Problem(objective=cp.Minimize(0), constraints=constraints)
    try:
        value = prob.solve(solver="MOSEK")
        
    except cp.error.SolverError as e:
        value = inf
        
    if return_all:
        R_vec_m = rho*vk_m - vkk_m - vec_combinations_m
        R_mat_m = rho*Vk_m - Vkk_m - mat_combinations_m

        R_vec_n = vk_n - a.T@fk - vec_combinations_n
        R_mat_n = Vk_n - Sk.T@A@Sk - mat_combinations_n
        return value, (P.value, p.value, A.value, a.value, duals_n.value, \
            duals_m.value, R_vec_m.value, R_mat_m.value, R_vec_n.value, R_mat_n.value)
    
    return value



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


def bisection_max_beta(beta_start, gamma, mu, L, rho, lyapunov_steps):
    beta_min = beta_start
    beta_max = 1.0
    while beta_max - beta_min > 1e-12:
        beta_next = (beta_max + beta_min) / 2
        value = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(beta_next, gamma, mu, L, rho, lyapunov_steps)

        if value != inf:
            beta_min = beta_next
        else:
            beta_max = beta_next
    
    return beta_min


def gamma_beta_pair_on_lyapunov_boundary(K, kappa):
    phi = np.cos( 2*np.pi / K )
    def gamma(beta):
        return kappa*(beta**2 - 2*(2*phi-1)*beta + 1) / (kappa*beta + 1)

    def func(beta):
        # now this is the function we want to solve the roots for
        q4 = (2*kappa**2 - kappa)*beta**4
        q3 = (-4*kappa**2*phi**2 - 2*kappa**2 + 4*kappa - 2)*beta**3
        q2 = (8*phi*kappa**2 - 2*kappa**2 + 8*kappa*phi**2 - 16*kappa*phi + 2*kappa + 8*phi - 2)*beta**2
        q1 = (-4*phi**2 - 2*kappa**2 + 4*kappa - 2)*beta
        q0 = -kappa + 2
        
        return q4 + q3 + q2 + q1 + q0

    sol = root_scalar(func, bracket=[0,1], method="brentq")
    beta = sol.root
    
    return gamma(beta)/mu, beta
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mu', '--mu', type=float)
    parser.add_argument('-K', '--K', type=float)
    parser.add_argument('-T', '--lyapunov_steps', type=int, default=1)
                            
    args = parser.parse_args()
    
    L = 1
    mu = args.mu
    kappa = mu/L
    K = args.K
    
    assert K >= 2
    
    gamma, beta = gamma_beta_pair_on_lyapunov_boundary(K, kappa)
    lyapunov_steps = args.lyapunov_steps
    # a version that guarantees to fill in the gap starting from smooth boundary
    if lyapunov_steps > 1:
        rho = 1.0
        max_beta = bisection_max_beta(beta, gamma, mu, L, rho, lyapunov_steps)
    else:
        max_beta = beta
        
    # try a gamma on the interior
    beta -= 0.2
    gamma -= 1
    
    # bisection to find the smallest rho
    rho_min = 0
    rho_max = 1
    while rho_max - rho_min > 1e-14:
            rho = (rho_max + rho_min) / 2
            value = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(max_beta, gamma, mu, L, rho, lyapunov_steps)

            if value != inf:
                rho_max = rho
            else:
                rho_min = rho
    
    rho = rho_max
    
    # gamma = 3.5
    # beta = 0.95
    # max_beta = beta
    # lyapunov_steps = args.lyapunov_steps
    # rho = 1.0
                
    print("mu    = ", mu)
    print("gamma = ", gamma)
    print("beta:                   ", beta)
    print("max beta for T=%d steps: " % lyapunov_steps, max_beta)
                
    print("smallest rho found", rho)
    value, diagnostics = hblyap.lyapunov_heavy_ball_momentum_multistep_smooth_boundary(max_beta, gamma, mu, L, rho, lyapunov_steps, return_all=True)
    print("\nOptimal value", value, "\n")

    P, p, A, a, duals_n, duals_m, R_vec_m, R_mat_m, R_vec_n, R_mat_n = diagnostics
        
    print("p\n", p)
    print("P\n", P)
    
    print("a\n", a)
    print("A\n", A)
    
    # print("Residual_n\n", R_vec_n)
    print("Residual_n\n", R_mat_n)
    # print(np.linalg.eigvalsh(R_mat_n))
    
    # print("Residual_m\n", R_vec_m)
    print("Residual_m\n", R_mat_m)
    
    R11_thr = (beta**2 * (-2*P[0,0]*(L - mu) * (beta**2 - rho) + L*p[0]*mu*(2 + beta*(2 + beta - 2 * rho) - rho)))/(2*(L-mu))
    R11_num = R_mat_m[0,0]
    
    print(R11_thr - R11_num)
    
    R12_thr = (beta*(2*P[0,0]*beta*(L - mu)*(beta**2 - rho) + L*p[0]*mu*(-((1 + beta)*(2 + beta + beta**2)) + 2*(1 + beta + beta**2)*rho)))/(2*(L-mu))
    R12_num = R_mat_m[0,1]
    
    print(R12_thr - R12_num)
    
    R55_thr = (p[0] - 2*P[3,3]*(L-mu))/(2*(L-mu))
    R55_num = R_mat_m[4,4]
    
    print(R55_thr - R55_num)
    
    # print(np.linalg.eigvalsh(R_mat_m))
    
    # print(VP.value)
    # quit()

    # d = p.value[1]
    # b = Pmat[0,3]
    # a = Pmat[0,0]
    # c = Pmat[3,3]

    # print("a       ", a)
    # print("b       ", b)
    # print("c       ", c)
    # print("d       ", d)
        
    # print("dual variables corresponding to MONOTONICITY constraints")
    # print(dual_m.value)
    
    # # look at residuals
    # matrix_combination_nonneg = sdp_prob.constraints[0]
    # vector_combination_nonneg = sdp_prob.constraints[1]
    # matrix_combination_monoto = sdp_prob.constraints[3]
    # vector_combination_monoto = sdp_prob.constraints[4]
    
    # # # print(VP_L_m)
    # Residual_m = (VP_L_m - VP_R_m).value
    # print("Rank of Residual_m: ", np.sum(np.linalg.svdvals(Residual_m)>1e-6))
    # print(np.linalg.svdvals(Residual_m))
    # # print(np.linalg.svdvals(Residual_m))
    
    # np.set_printoptions(5)
    # print(Residual_m)


    # p = p.value[0]
    # etak = dual_m.value[0]
    # etas = dual_m.value[-1]
    # print(etak, p*rho/(L-mu))
    # print(etas, p*(1-rho)/(L-mu))
    
    # b = Pmat[2,2]
    # c = Pmat[1,2]
    # print(b / (1-beta) * (L-mu))
    # print(c * rho)
    
    # eigvals, eigvecs = np.linalg.eigh(-Residual_m)
    # # print(eigvals)
    # # print(eigvecs)
    # # # v1 = (eigvecs[:,-1]*np.sqrt(eigvals[-1]))
    # # # v2 = (eigvecs[:,-2]*np.sqrt(eigvals[-2]))
    # # # v3 = (eigvecs[:,-3]*np.sqrt(eigvals[-3]))

    # A = a*2*(L-mu)
    # B = b*2*(L-mu)

    # eps = 3e-9
    # U = np.linalg.cholesky(-Residual_m + eps*np.eye(5), upper=True)
    # print("U")
    # print(U)
    
    # z = 2*(L-mu)
    
    # U[3,:] = U[3,:] - U[4,:]*U[3,4]/U[4,4]
    # U[0,:] = U[0,:] - U[4,:]*U[0,4]/U[4,4]
    # U[0,:] = U[0,:] - U[3,:]*U[0,3]/U[3,3]
    # print(U)
    
    # print(U[0,0]**2)
    # print((A*(1-beta**2) + beta**2*mu*L)/z)
    
    # print(U[0,0]*U[0,3])
    # print((B - A*gamma*beta + (L*gamma-1)*mu*beta)/z)
    
    # print(U[0,0]*U[0,4])
    # print((L-B)*beta/z)
    
    # print(U[0,0]*(U[0,3]+U[0,4]))
    # print((B*(1-beta) + beta*(L-mu) - gamma*beta*(A-mu*L))/z)
    
    # print("check")
    # print(B)
    # print(beta*(A*gamma- (L*gamma-1)*mu))
    # print("\n")
    # print(((B-L)*beta)/(2*(L-mu)))
    # print(-U[0,0]*U[0,4])
    
    # print("\n")
    # print((1+gamma*(B-L))/(2*(L-mu)))
    # print(U[3,3]*U[3,4] + U[0,3]*U[0,4] )
    
    # print("\n")
    # print((beta-2 + A*gamma**2 - gamma*(L*gamma-2)*mu)/(2*(L-mu)))
    # print(U[0,3]**2 + U[3,3]**2)
    
    # print("\n")
    # print(beta/(2*(L-mu)))
    # print(U[0,4]**2 + U[3,4]**2 + U[4,4]**2)
    
    # print("Cholesky upper")
    # print(U)
    # # ratio = U[0,0]/U[1,1]
    # # U[1,:] *= ratio
    # # U[1,:] += U[0,:]
    # # print("after row op")
    # # print(U)
    
    
    

    
    # eigv = eigvals[-1]
    # eigvec = eigvecs[:,-1]
    # v1 = eigvec[0]
    # v4 = eigvec[3]
    # v5 = eigvec[4]


    # t1 = gamma**3
    # t2 = 3*gamma**2*(1-L*gamma)
    # t3 = gamma*(beta**2*(L*gamma-1) + 3*(L*gamma-1)**2 - mu*gamma*beta**3 - beta*(L*mu*gamma**2 - 2*mu*gamma + 2))
    # t4 = -((L*gamma-1+beta**2)*(beta**2 + (L*gamma-1)**2 - beta*(L*mu*gamma**2 - 2*mu*gamma + 2)))
    
    # roots = np.sort(np.roots([t1, t2, t3, t4]))
    # roots_pos = roots[roots>=0]
    
    # test_b = roots_pos[0] / (2*(L-mu))
    # test_a = - (beta**2 - beta*(mu*L*gamma**2-2*mu*gamma+2) + (1+gamma*(b*2*(L-mu)-L))**2) / (2*(L-mu)*beta*gamma**2)
    
    # print(a - test_a)
    # print(b - test_b)
    
    
    # ahat = a*2*(L-mu)
    # ahatRHS = - (beta**2 - beta*(mu*L*gamma**2-2*mu*gamma+2) + ) / (beta*gamma**2)
    # bhat = b*2*(L-mu)
    # LHS = (beta**2*mu*L - ahat*(beta**2-1))*(1+gamma*(bhat-L))
    # RHS = beta*(2*b*(L-mu)-L)*(2*b*(L-mu) + mu*beta*(L*gamma-1) - 2*a*gamma*beta*(L-mu))
    # RHS = beta*(bhat-L)*(bhat + mu*beta*(L*gamma-1) - ahat*gamma*beta)
    # print(LHS- RHS)
    
    # print(v1, v4, v5, np.sqrt(eigv))
    
    # print("v4v5")
    # print(v4*v5*eigv*(-2))
    # print( (1+(2*b-1)*L*gamma-2*b*gamma*mu)/(L-mu) )
    
    # print("v4v4")
    # print(v4*v4*eigv*(-1))
    # print( (-2+beta+2*a*gamma**2*(L-mu)+2*gamma*mu-gamma**2*mu*L) /(2*(L-mu)) )
    
    # print("v5v5")
    # print(v5*v5*eigv*(-1))
    # print(-beta/(2*(L-mu)))
    
    # print("does THIS equal to 1")
    # print(((-2*a*(-1 + beta**2)*(L - mu) + L*beta**2*mu)*(1 + (-1 + 2*b)*L*gamma - 2*b*gamma*mu))/(beta*((-1 + 2*b)*L - 2*b*mu)*(2*b*(L - mu) + beta*(-1 + L*gamma)*mu + 2*a*beta*gamma*(-L + mu))))
    
    
    # print("does it equal to 1")
    # equalto1 = -( beta*(beta-2+2*a*(L-mu)*gamma**2 + 2*gamma*mu - L*mu*gamma**2) )/( (1+(2*b-1)*L*gamma - 2*b*gamma*mu)**2 )
    # print(equalto1)
    
    # print(eigv*v5**2)
    # print(-(L-mu-1)*2*(L-mu)**2 )
    # print(np.linalg.cholesky(Residual_m_principal))
    
    
    # print(sdp_prob.constraints[3].dual_value)
    
    # print(np.trace(Residual_m@sdp_prob.constraints[3].dual_value))
    # print(R45, -(b*gamma + (1-gamma)/(2*(L-mu))))
    # print(R15, beta*(b - L/(2*(L-mu))))
    # print(R45, b*gamma - (gamma*L-1)/(2*(L-mu)))

    # print(beta, (1-c*2*(L-mu)))
    # print(beta*mu, (1-b*2*(L-mu)/L)*beta)
    # print(c, 1/(2*(L-mu))- R55)
    # print(b, L/(2*(L-mu)) - R15/beta)
    # print(a, 1/(beta**2-1)*((beta**2)*mu*L/(2*(L-mu)) - R11))
        
    # print("Singular values of Residual_m")
    # print(np.linalg.svdvals(Residual_m))
    
    # residual_m = (Vp_R_m - Vp_L_m).value
    # print(residual_m)
    
    # lambdakk1 = dual_m.value[4]
    # print("lambda(k, k+1)", lambdakk1)
    # print("d/(1-mu)      ", d/(1-mu))
    
    # mm = lyapunov_steps+3
    # M = np.zeros((mm, mm))
    # row_idx, col_idx = np.where(~np.eye(lyapunov_steps+3, dtype=bool))
    # M[row_idx, col_idx] = dual_m.value
    
    # fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    # ax.imshow(M)
    
    # headers = [r"$k-1$", r"$k$"]
    # for t in range(1,lyapunov_steps+1):
    #     headers += [r"$k+%d$"%t]
    # headers += [r"*"]
    
    # ax.set_xticks(np.arange(len(headers))[::2])
    # ax.set_yticks(np.arange(len(headers))[::2])
    # ax.set_xticklabels(headers[::2], rotation=45)
    # ax.set_yticklabels(headers[::2])
    # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    # ax.set_title(r"$T=%d, \, \mu=%.2f$" % (lyapunov_steps, mu), fontsize=17)

    # fig_fn = "tmp/dual_m_mu=%.2f_T=%d_K=%.2f.png" % (mu, lyapunov_steps, K)
    # plt.savefig(fig_fn)
    # print("Figure saved at \n%s" % fig_fn)

    # np.set_printoptions(precision=4)
    # print("dual variables corresponding to MONOTONICITY constraints")
    # print(M)
    
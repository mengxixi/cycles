import algorithms.heavy_ball.lyapunov as hblyap
import numpy as np

mu = 0.9
L = 1

beta = 0.5
gamma = 1

lyapunov_steps = 1
rho = 1 # 0.587717609

sdp_prob, P, p, dual_n, dual_m = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, lyapunov_steps, return_all=True)
print("P=")
print(np.triu(P.value))
print("p=")
print(p.value)
print("dual variables corresponding to NONNEGATIVITY constraints")
print(dual_n.value)
print("dual variables corresponding to MONOTONICITY constraints")
print(dual_m.value)

if P.value is not None:
    print("eigvals of P")
    eigvals = np.linalg.eigvalsh(P.value)
    print(eigvals)
    print("rank of P", np.linalg.matrix_rank(P.value))

# Try to clean up a bit

# P.value[2,:] = 0
# P.value[:,2] = 0

# ind = [0, 1, 2, 3, 5, 6, 7, 8, 9]
# dual_m.value[ind] = 0

# is_feasible = True
# for constraint in sdp_prob.constraints:
#     violation = constraint.violation()
#     if np.any(violation > 1e-6):
#         is_feasible = False
#         print(constraint)
#         break

# print("Feasible after clean up = ", is_feasible)
    




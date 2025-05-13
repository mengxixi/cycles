import algorithms.heavy_ball.lyapunov as hblyap
import numpy as np

# mu = 0.1
# L = 1

# beta = 0.52836725292
# gamma = 2.5


mu = 0.3 #0.75
L = 1

beta = 0.6118306605 # 0.84996902 
gamma = 2.5

# mu = 0.5
# L = 1

# beta = 0.92652155
# gamma = 1

lyapunov_steps = 1
rho = 1 # 0.587717609

value, sdp_prob, P, p, dual_n, dual_m = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, lyapunov_steps, return_all=True)
print("\n", value, "\n")

print("P =")
print(np.triu(P.value))

Pmat = P.value
p1 = p.value[1]
b = Pmat[0,3]
a = Pmat[0,0]
c = Pmat[3,3]

print("a       ", a)
print("b       ", b)
print("c       ", c)
# print("sqrt(ac)", np.sqrt(a*c)) 

# print("tr(P)=", np.linalg.trace(P.value))

print("p =", p.value)
# print("dual variables corresponding to NONNEGATIVITY constraints")
# print(dual_n.value)
# print("dual variables corresponding to MONOTONICITY constraints")
# print(dual_m.value)

# if P.value is not None:
#     print("eigvals of P")
#     eigvals = np.linalg.eigvalsh(P.value)
#     print(eigvals)
#     print("rank of P", np.linalg.matrix_rank(P.value))

# # Try to see if Ghadimi's thing works
# b = 1/gamma - L/2

# p1 = np.maximum(beta, beta - 1 + 2*beta*(beta-gamma*mu)/(2-gamma*L))

# p_fix = np.array([p1, 1])
# P_fix = np.array([
#     [b, -b, 0, 0],
#     [-b, b, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0]
# ])
# P_fix = None

# value = hblyap.lyapunov_heavy_ball_momentum_multistep_fixed(beta, gamma, mu, L, rho)
# print("\n")
# print(value, "\n")
# print("P=")
# print(np.triu(P.value))
# print("b =", b)
# print("p=")
# print(p)
# print("dual variables corresponding to NONNEGATIVITY constraints")
# print(dual_n.value)
# print("dual variables corresponding to MONOTONICITY constraints")
# print(dual_m.value)

# is_feasible = True
# for constraint in sdp_prob.constraints:
#     violation = constraint.violation()
#     if np.any(violation > 1e-6):
#         is_feasible = False
#         print(constraint)
#         break

# print("Feasible after clean up = ", is_feasible)
    




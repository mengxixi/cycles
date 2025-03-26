import algorithms.heavy_ball.lyapunov as hblyap
import numpy as np

mu = 0.1
L = 1

beta = 0.5
gamma = 2.55

lyapunov_steps = 2
rho = 1.0


sdp_prob, P, p = hblyap.lyapunov_heavy_ball_momentum_multistep(beta, gamma, mu, L, rho, lyapunov_steps, return_all=True)
print(P.value)
print(p.value)

if P.value is not None:
    print("eigvals of P")
    eigvals = np.linalg.eigvalsh(P.value)
    print(eigvals)



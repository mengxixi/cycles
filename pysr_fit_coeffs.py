import os
import argparse
import pickle 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from pysr import PySRRegressor, TemplateExpressionSpec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": "Helvetica",
    "figure.dpi": 600
})

TMP_DIR = "tmp"
LOG_DIR = "logs"


def get_gamma_beta_pair(kappa, K):
    phi = np.cos( 2*np.pi /  K )
    
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

    return gamma(beta), beta


def load_xy(coeff_label):
    coeff_fn = os.path.join(LOG_DIR, "")
    coeff_fn = os.path.join(LOG_DIR, "lyap_param_vs_both_K_and_mu.pkl")
    with open(coeff_fn, "rb") as f:
        coeff_dict = pickle.load(f)
        
        Ks = coeff_dict["K"]
        ind_K = np.where(Ks >= 4)[0]
        Ks = Ks[ind_K]
        
        phis = np.cos(2*np.pi/Ks)
        mus = coeff_dict["mu"][ind_K]
        coeffs = coeff_dict[coeff_label][ind_K]
        
        betas = []
        gammas = []
        for mu, K in zip(mus, Ks):
            kappa = mu # L=1
            beta = get_gamma_beta_pair(kappa, K)[1]
            gamma = (1-beta)/((1-beta*kappa)**2)
            gamma *= (1+3*beta*(1-kappa)-kappa*beta**2 + np.sqrt(4*beta*(2-kappa-kappa*beta)*(1+beta-2*beta*kappa)) )
            betas += [beta]
            gammas += [gamma]
            
        gammas = np.array(gammas)
        betas = np.array(betas)
        
        _mu = mus[5]
        ind_mu = np.where(mus == _mu)[0]

        # filter again everything we may possibly need
        mus = mus[ind_mu]
        coeffs = coeffs[ind_mu]
        betas = betas[ind_mu]
        gammas = gammas[ind_mu]
        phis = phis[ind_mu]
        
        xs = [
            gammas,
            betas,
            # mus,
        ]        
        
        xs = np.array(xs).T
        y = coeffs
        
    return xs, y


def pysr_fit(X, y):
    # Create template that combines f(x1, x2) and g(x3):
    expression_spec = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x0", "x1"],
        parameters={"p": 4},
        combine="(((x1*p[1] - p[2]) * x0) + x0 + x1*p[3] ) + p[4]+ 0*f(x1)"
    )
    
    model = PySRRegressor(
        populations=16,
        population_size=50,
        maxsize=25,
        # maxdepth=10,
        precision=64,
        niterations=10000,  # < Increase me for better results
        # expression_spec=expression_spec,
        binary_operators=["+", "*", "/"],
        unary_operators=[
            # "square(x) = x^2",
            # "cube(x) = x^3",
            # "fourth(x) = x^4",
            # "cos",
            # "exp",
            # "sin",
            "inv(x) = 1/x",
            # "sqrt" ,
            # ^ Custom operator (julia syntax)
        ],
        complexity_of_constants=2,
        turbo=True,
        nested_constraints={
                # "square": {"sqrt": 0},
                # "sqrt": {"square": 0},
            },
        extra_sympy_mappings={"inv": lambda x: 1 / x, 
                              "square": lambda x: x**2,
                              "cube": lambda x: x**3,
                            #   "fourth": lambda x: x**4,
                              },
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = 0.5*(prediction - target)^2",
        # elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
    )
    model.fit(X, y)
    return model
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coef', type=str, choices=["a", "b", "c", "d", "e", "f", "p1", "p2"])
    args = parser.parse_args()
    
    coeff_label = args.coef
    
    X, y = load_xy(coeff_label)
    
    # kappa = mu # L=1
    # beta_min = ((2-kappa) - np.sqrt( (1-kappa)*(5-kappa) )) / (2*kappa-1)
    # X = np.linspace(beta_min, 1, num=100)[:,None]
    # y = (1-X)/((1-X*kappa)**2)
    # y *= (1+3*X*(1-kappa)-kappa*X**2 + np.sqrt(4*X*(2-kappa-kappa*X)*(1+X-2*X*kappa)) )
    
    model = pysr_fit(X,y)
    print(model)
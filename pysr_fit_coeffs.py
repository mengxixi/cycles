import os
import argparse
import pickle 

import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor, TemplateExpressionSpec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": "Helvetica",
    "figure.dpi": 600
})

TMP_DIR = "tmp"
LOG_DIR = "logs"


def load_xy(mu, hb_param_labels, coeff_label):
    coeff_fn = os.path.join(LOG_DIR, "lyap_param_mu=%.2f_%s.pkl" % (mu, coeff_label))
    with open(coeff_fn, "rb") as f:
        coeff_dict = pickle.load(f)
        
        xs = []
        for label in hb_param_labels:
            xs.append(coeff_dict[label])
        
        xs = np.array(xs).T
        y = coeff_dict["coeffs"]
        
    return xs, y


def pysr_fit(X, y):
    # Create template that combines f(x1, x2) and g(x3):
    expression_spec = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x1", "x2"],
        parameters={"p": 2},
        combine="1/(p[1]*x1*x2 + p[2]*x1 + 2) + 0*f(x1)"
    )
    
    model = PySRRegressor(
        populations=16,
        population_size=100,
        maxsize=15,
        # maxdepth=10,
        precision=64,
        niterations=1000,  # < Increase me for better results
        expression_spec=expression_spec,
        binary_operators=["+", "*", "/"],
        unary_operators=[
            # "cos",
            # "exp",
            # "sin",
            # "inv(x) = 1/x",
            # "sqrt" ,
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
    )
    model.fit(X, y)
    return model
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mu', '--mu', type=float)
    parser.add_argument('-x', '--xlabs', type=str, choices=["gammas", "betas"], nargs='+')
    parser.add_argument('-y', '--ylab', type=str, choices=["a", "b", "c", "d", "e", "f", "p1", "p2"])
                            
    args = parser.parse_args()
    
    mu = args.mu
    hb_param_labels = args.xlabs
    coeff_label = args.ylab
    
    X, y = load_xy(mu, hb_param_labels, coeff_label)
    
    # kappa = mu # L=1
    # beta_min = ((2-kappa) - np.sqrt( (1-kappa)*(5-kappa) )) / (2*kappa-1)
    # X = np.linspace(beta_min, 1, num=100)[:,None]
    # y = (1-X)/((1-X*kappa)**2)
    # y *= (1+3*X*(1-kappa)-kappa*X**2 + np.sqrt(4*X*(2-kappa-kappa*X)*(1+X-2*X*kappa)) )
    
    model = pysr_fit(X,y)
    print(model)
    
    ypred = model.predict(X)
    
    nrows = 1
    ncols = len(hb_param_labels)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(2.5*ncols,2.5*nrows),
                            constrained_layout=True)
    
    for j, xlab in enumerate(hb_param_labels):
        ax = axs if ncols == 1 else axs[j]
        
        ax.plot(X[:,j], y, label=r"$%s$" % coeff_label, color="navy")
        ax.plot(X[:,j], ypred, label=r"$\hat{%s}$" % coeff_label, color="skyblue", linestyle="--")
    
        x_ax_lab = r"$\gamma$" if xlab == "gammas" else r"$\beta$"
        ax.set_xlabel(x_ax_lab, fontsize=17)
        ax.set_ylabel("coeff", fontsize=17)
        
    figname = "pysr_lyap_param_mu=%.2f_%s.png" % (mu, coeff_label)
    fig_fn = os.path.join(TMP_DIR, figname)
    fig.savefig(fig_fn)
    # plt.savefig(fig_fn.replace("png", "pdf"))
    print("Figure saved at \n%s" % fig_fn)

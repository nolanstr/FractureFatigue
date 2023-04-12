import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bingo.symbolic_regression.agraph.agraph import AGraph

from matplotlib.backends.backend_pdf import PdfPages

R = 0.5
C = 9E-8
m = 3

def plot_comparison(model, features, test_datasets, FIG_NAME="TEST"):

    pp = PdfPages(FIG_NAME + ".pdf")

    def plot_on_pdf(feature, test_data):
        
        delta_K = model.evaluate_equation_at(feature).flatten() * 1.0998 * (1-R)
        approx_dadN = C * pow(delta_K, m) 
        aw_feature = feature[:,1]
        
        test_aw = test_data[:,0]
        test_dadN = test_data[:,1]

        fig, axis = plt.subplots()
        axis.set_yscale('log')
        axis.scatter(test_aw, test_dadN, color='k', label="Experimental Data")
        idxs = np.where(aw_feature>test_aw.min())
        axis.plot(aw_feature[idxs], approx_dadN[idxs], color='b', label="Aprox. Solution")

        axis.set_xlabel(r"$\frac{a}{w}$")
        axis.set_ylabel(r"$\frac{da}{dN}$")
        axis.legend(loc="upper right")

        fig.tight_layout()
        pp.savefig(fig, dpi=1000, transparent=True)
    
    for feature, test_data in zip(features, test_datasets):
        plot_on_pdf(feature, test_data)
    
    pp.close()


if __name__ == "__main__":
    string = "(0.021864613120989473)((X_2)(0.12817956686642362 + X_1 + (X_1)(X_1) + (9.114380860565196e-05)(((X_1)((X_1)(X_1)))(-2920.2841210194524 + X_0 - (X_1 + X_2)))) + exp((X_1)((X_1)(X_1))))" 
    model = AGraph(equation=string)
    features = [np.load(f"../test_data/Al{i}.npy") for i in [2,3]]
    test_datasets = [np.load(f"../test_data/Al{i}_test_data.npy") for i in [2,3]]

    plot_comparison(model, features, test_datasets)

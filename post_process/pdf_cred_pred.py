import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 16})
import numpy as np
import glob

from bingo.symbolic_regression.agraph.agraph import AGraph

X_idx = np.array([0,1,7])
y_idx = np.array([2])

E_dict = {"29700":"Steel AISI 4130",
          "10200":"Aluminum 6061"}

def make_pdf(model, datasets, FIG_NAME="base_pdf"):

    pp = PdfPages(FIG_NAME+".pdf")

    def plot_on_pdf(dataset):

        fig, axis = plt.subplots()
        X = dataset[0] 
        approx = model.evaluate_equation_at(X).flatten()
        y = dataset[1].flatten()
        aw = X[:,1].flatten()
        axis.scatter(aw, y, color='k', label="Simulated Data")
        axis.plot(aw, approx, color=plt.cm.tab10(0), label="Approx. Solution")
        axis.legend(loc="upper left")
        title = f"{E_dict[str(int(X[0,0]))]}, E = {X[0,0]}, P = {round(X[0,2],2)}"
        axis.set_xlabel(r"$\frac{a}{w}$")
        axis.set_ylabel(r"K")
        fig.suptitle(title)
        plt.tight_layout()
        pp.savefig(fig, dpi=1000, transparent=True)

    for dataset in datasets:
        plot_on_pdf(dataset)
    
    pp.close()

def get_data(count=10):

    files = glob.glob("../npy_files/*.npy")
    datasets = [np.load(f) for f in files]
    subsets = []

    for dataset in datasets:
        P_vals = np.unique(dataset[:,7])
        idxs = [np.where(dataset[:,7]==P_val)[0] for P_val in P_vals]
        
        picks = np.linspace(3, len(idxs)-4, count, dtype=int)
        
        for pick in picks:
            if np.unique(dataset[idxs[pick],:][:,1]).shape[0] > 4:
                subsets += [[dataset[idxs[pick],:][:,X_idx],
                        dataset[idxs[pick],:][:,y_idx]]]
    for subset in subsets:
        print(np.unique(subset[0][:,1]))
    return subsets

if __name__ == "__main__":

    string = "(0.02227483155596123)((X_2)(0.12375640008024923 + X_1 + (X_1)(X_1) + (8.425982109384921e-05)(((X_1)((X_1)(X_1)))(X_0 + (-2.273586491247473)(X_2) - (X_1)))) + exp((X_1)((X_1)(X_1))))" 
    model = AGraph(equation=string)
    str(model)
    datasets = get_data() 
    make_pdf(model, datasets)

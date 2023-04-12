import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import pickle


def grab_and_organize_population(DIR):

    files = glob.glob(DIR+"/*.pkl")
    files, gens = organize_files(files)
    pops = []
    
    for FILE in files:
        f = open(FILE, "rb")
        pops.append(pickle.load(f))
        f.close()

    return pops, gens

def organize_files(files):
    gens = []
    for FILE in files:
        num_val = int(''.join([i for i in FILE if i.isdigit()]))
        gens.append(num_val)

    files = [files[i] for i in np.argsort(gens)]
    gens = [gens[i] for i in np.argsort(gens)]

    return files, np.array(gens)

def get_fitness(pops):
    
    fits = np.empty((len(pops), len(pops[0])))
    
    for i, pop in enumerate(pops):
        fits[i] = np.array([ind.fitness for ind in pop])

    return fits

def organize_pop_i(pops, i):

    pop_i = []
    for pop in pops:
        pop_i += pop[i]
    return pop_i

def get_n_best_from_pop(pop, fits, n=1):
    
    sorted_idxs = np.argsort(fits)
    inds = [pop[i] for i in sorted_idxs[:n]]

    return inds

if __name__ == "__main__":
    
    DIR = sys.argv[1]
    pops, gens = grab_and_organize_population(DIR)
    pops, gens = pops[1:], gens[1:] #first population not useful here
    fits = []
    plot_chains = len(pops[0])
    if len(sys.argv) == 3:
        n_best = int(sys.argv[2])
    else:
        n_best = 5
    compressed_pops = [organize_pop_i(pops, i) for i in range(len(pops[0]))]
    compressed_fits = [[i.fitness for i in pop] for pop in compressed_pops]

    best_n_from_each_trace = [get_n_best_from_pop(pop, fits) for pop, fits in \
                                        zip(compressed_pops, compressed_fits)]
    
    best_fits = [i[0].fitness for i in best_n_from_each_trace]

    for i in np.argsort(best_fits):
        trace_models = best_n_from_each_trace[i]
        for model in trace_models:
            print(f"Model: {str(model)}")
            print(f"fitness: {model.fitness}")

    import pdb;pdb.set_trace()

import numpy as np
import time
import copy
import pickle
import sys
from multiprocessing import Pool

from bingo.symbolic_regression import ExplicitTrainingData,\
                                      ExplicitRegression
from bingo.local_optimizers.continuous_local_opt import \
                                      ContinuousLocalOptimization
from bingo.evolutionary_optimizers.get_all_arch import *

from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function import \
                                      BayesFitnessFunction

from bingo.evolutionary_optimizers.evolutionary_optimizer import \
                        load_evolutionary_optimizer_from_file as leoff


def get_pickles(DIR):

    files = glob.glob(DIR + "/*.pkl")

    try:
        pickles = [leoff(f).island for f in files]
    except:
        pickles = [leoff(f) for f in files]
    gens = np.array([p.generational_age for p in pickles])
    idxs = np.argsort(gens)

    islands = [pickles[i] for i in idxs]
    gens = gens[idxs]
    fits = np.array([[ind.fitness for ind in island.population] \
                                                        for island in islands])
    comps = np.array([[ind.get_complexity() for ind in island.population] \
                                                        for island in islands])

    unique_comps = np.unique(comps)
    models = []

    for comp in unique_comps:
        rows, cols = np.where(comps==comp)
        idx = np.nanargmin(fits[rows,cols])
        models.append(islands[rows[idx]].population[cols[idx]])
    
    fits = [ind.fitness for ind in models]

    for idx in np.argsort(fits):
        ind = models[idx]
        print(str(ind))
        print(f"-NMLL: {ind.fitness}, Complexity: {ind.get_complexity()}")


if __name__ == "__main__":
    DIR = sys.argv[-1]
    if '' in DIR.split('/'):
        fig_name = DIR.split('/')[-2]
    else:
        fig_name = DIR.split('/')[-1]
    get_pickles(DIR)

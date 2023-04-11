import numpy as np
import time
import copy
import pickle

from bingo.symbolic_regression import ExplicitTrainingData,\
                                      ExplicitRegression
from bingo.evolutionary_optimizers.get_all_arch import *

from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function import \
                                      BayesFitnessFunction
from bingo.local_optimizers.cust_continuous_local_opt import \
                                      ContinuousLocalOptimization

def make_training_data():
    
    num_points = np.minimum(np.array(num_points), np.array([n]*len(num_points))) 
    training_data = ExplicitTrainingData(x, y)

    return training_data, num_points

def get_cred_pred_intervals_on_subsets(model, n=100):

    training_data, num_points = make_training_data(n)
    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, num_points, algorithm='lm')

    smc_hyperparams = {'num_particles':5000,
                       'mcmc_steps':10,
                       'ess_threshold':0.75}
    multisource_info = num_points
    random_sample_info = None

    bff = BayesFitnessFunction(local_opt_fitness,
                                   smc_hyperparams=smc_hyperparams,
                                   multisource_info=multisource_info,
                                   random_sample_info=random_sample_info)
    n_linspace = 1000
    inputs = []
    models = []
    nmll, step_list, _ = bff(model, return_nmll_only=False)
    print(f'-nmll = {nmll}')
    print(f'model = {str(model)}')

    inputs = [bff.estimate_cred_pred(model, step_list, subset=i, \
                                  linspace=1000) for i in range(len(num_points))]
    import pdb;pdb.set_trace()
    inputs = [i[:-1] + [i[-2]] for i in inputs]
    
    return inputs, bff

if __name__ == "__main__":
    string = "" 
    model = AGraph(equation=string)

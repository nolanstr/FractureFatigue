import numpy as np
import time
import copy
import pickle
import glob
import matplotlib.pyplot as plt

from multiprocessing import Pool

from bingo.symbolic_regression import ExplicitTrainingData,\
                                      ExplicitRegression
from bingo.local_optimizers.continuous_local_opt import \
                                      ContinuousLocalOptimization

from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness.ensemble_bayes_fitness_function import \
                                     EnsembleBayesFitnessFunction 

import sys;sys.path.append('../');sys.path.append('../../')

from model_sampler import MCMCSampleModel
from population_manager import Population
from old_scripts.sample_models import SampleModels

from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function import \
                                      BayesFitnessFunction

STACK_SIZE = 24

def make_training_data():
    
    files = glob.glob("./*.npy")
    data_sets = [np.load(FILE) for FILE in files]
    num_points = tuple([data_set.shape[0] for data_set in data_sets])
    data = np.vstack(data_sets)
    x, y = data[:,np.array([0,1,7])], data[:,2].reshape((-1,1))
    training_data = ExplicitTrainingData(x, y)

    return training_data, num_points

def make_bff(training_data, num_points):

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')

    smc_hyperparams = {'num_particles':20,
                       'mcmc_steps':5,
                       'ess_threshold':0.75}
    multisource_info = num_points
    random_sample_info = 50

    bff = BayesFitnessFunction(local_opt_fitness,
                                   smc_hyperparams=smc_hyperparams,
                                   multisource_info=multisource_info,
                                   random_sample_info=random_sample_info)
    return bff 

def make_needed_components():

    training_data, num_points = make_training_data()
    bff = make_bff(training_data, num_points)

    component_generator = ComponentGenerator(training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    #component_generator.add_operator("/")
    component_generator.add_operator("exp")
    component_generator.add_operator("log")
    #component_generator.add_operator("pow")
    component_generator.add_operator("sqrt")
    #component_generator.add_operator("sin")
    #component_generator.add_operator("cos")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True
                                       )
    
    return bff, crossover, mutation, generator


if __name__ == "__main__":
    
    replace_prob = np.exp
    bff, crossover, mutator, generator = make_needed_components()
    
    mcmc_sampler = MCMCSampleModel(bff, mutator, replace_prob, iters=100,
            trace=True)
    POP_SIZE = 40
    GENS = 200

    evolver = Population(POP_SIZE, generator, replace_prob, mcmc_sampler)
    
    final_population = evolver.evolve_population(GENS, nprocs=24,
            track_percent=None)

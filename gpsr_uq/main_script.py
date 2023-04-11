# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pickle
import glob

from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.selection.bayes_crowding import BayesCrowding
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.evolutionary_optimizers.parallel_archipelago import \
                                            ParallelArchipelago, \
                                            load_parallel_archipelago_from_file
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
import sys
sys.path.append('../');sys.path.append('../../');sys.path.append('../../../')
sys.path.append('../../../../')
from research.GenerateSeeds import SubgraphSeedGenerator
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.bayes_fitness.ensemble_bayes_fitness_function \
                                        import BayesFitnessFunction

POP_SIZE = 104
STACK_SIZE = 48
MAX_GEN = 10000
FIT_THRESH = -np.inf
CHECK_FREQ = 50
MIN_GEN = 500

def make_training_data():
    
    files = glob.glob("../npy_files/*.npy")
    data_sets = [np.load(FILE) for FILE in files]
    num_points = tuple([data_set.shape[0] for data_set in data_sets])
    data = np.vstack(data_sets)
    x, y = data[:,np.array([0,1,7])], data[:,2].reshape((-1,1))
    training_data = ExplicitTrainingData(x, y)

    return training_data, num_points

def execute_generational_steps():

    training_data, num_points = make_training_data()

    component_generator = ComponentGenerator(training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")
    component_generator.add_operator("exp")
    component_generator.add_operator("log")
    component_generator.add_operator("pow")
    component_generator.add_operator("sqrt")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True
                                       )

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    smc_hyperparams = {'num_particles':10,
                       'mcmc_steps':10,
                       'ess_threshold':0.75}
    multisource_info = num_points
    random_sample_info = 50

    bff = BayesFitnessFunction(local_opt_fitness,
                                   smc_hyperparams,
                                   multisource_info,
                                   random_sample_info,
                                   ensemble=5)

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    evaluator = Evaluation(bff, redundant=True, multiprocess=40)

    selection_phase=BayesCrowding()
    ea = GeneralizedCrowdingEA(evaluator, crossover,
                      mutation, 0.4, 0.4, selection_phase)


    island = Island(ea, agraph_generator, POP_SIZE, hall_of_fame=pareto_front)
    opt_result = island.evolve_until_convergence(max_generations=MAX_GEN,
                                                  fitness_threshold=FIT_THRESH,
                                        convergence_check_frequency=CHECK_FREQ,
                                              checkpoint_base_name='checkpoint')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == '__main__':
    execute_generational_steps()


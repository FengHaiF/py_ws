import numpy as np

from optimize.nd_sort import nd_sort
from optimize.crowding_distance import crowding_distance
from optimize.tournament import tournament
from optimize.environment_selection import environment_selection
from optimize.GLOBAL import Global

import time

from Regresser import Ensemble
import warnings
warnings.filterwarnings(action='ignore', \
             module='sklearn')

ensemble = Ensemble()
ensemble.load_model()

Global = Global()
Global.load_regress(ensemble)

class nsgaii(object):
    """
    NSGA-II algorithm
    """

    def __init__(self, decs=None, ite=100, eva=100 * 20):
        self.decs = decs
        self.ite = ite
        self.eva = eva

    def run(self):
        start = time.clock()
        if self.decs is None:
            population = Global.initialize()
        else:
            population = Global.individual(self.decs)

        front_no, max_front = nd_sort(population[1], np.inf)
        crowd_dis = crowding_distance(population[1], front_no)
        evaluation = self.eva
        while self.eva >= 0:
            fit = np.vstack((front_no, crowd_dis)).T
            mating_pool = tournament(2, Global.N, fit)
            pop_dec, pop_obj = population[0], population[1]
            parent = [pop_dec[mating_pool, :], pop_obj[mating_pool, :]]
            offspring = Global.variation(parent[0],boundary=(Global.lower,Global.upper))
            population = [np.vstack((population[0], Global.individual(offspring)[0])), np.vstack((population[1], Global.individual(offspring)[1]))]
            population, front_no, crowd_dis,_ = environment_selection(population, Global.N)
            self.eva = self.eva - Global.N
            if self.eva%(10*evaluation/self.ite) == 0:
                end = time.clock()
                print('Running time %10.2f, percentage %s'%(end-start,100*(evaluation-self.eva)/evaluation))
        print("=========================================")    
        return population

nsga = nsgaii()  
pop = nsga.run()
# print(pop)
# pop_obj = pop[1]
# front_no, max_front = nd_sort(pop_obj, 1)

optimal_policy = pop[0][-1]
optimal_result = pop[1][-1]
print("optimal_policy : ",optimal_policy," || optimal_result: ",optimal_result)
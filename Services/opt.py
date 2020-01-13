import numpy as np

from optimize.nd_sort import nd_sort
from optimize.crowding_distance import crowding_distance
from optimize.tournament import tournament
from optimize.environment_selection import environment_selection
from optimize.optset import Optset

import time

from Regresser import Ensemble
import warnings
warnings.filterwarnings(action='ignore', \
             module='sklearn')



# Global = Global()
# Global.load_regress(ensemble)

class nsgaii(object):
    """
    NSGA-II algorithm
    """

    def __init__(self, decs=None, ite=100, eva=100 * 20):
        self.decs = decs
        self.ite = ite
        self.eva = eva

    def setGlobal(self,glo_set):
        self.Global = glo_set

    def run(self):
        start = time.clock()
        if self.decs is None:
            population = self.Global.initialize()
        else:
            population = self.Global.individual(self.decs)

        front_no, max_front = nd_sort(population[1], np.inf)
        crowd_dis = crowding_distance(population[1], front_no)
        evaluation = self.eva
        while self.eva >= 0:
            fit = np.vstack((front_no, crowd_dis)).T
            mating_pool = tournament(2, self.Global.N, fit)
            pop_dec, pop_obj = population[0], population[1]
            parent = [pop_dec[mating_pool, :], pop_obj[mating_pool, :]]
            offspring = self.Global.variation(parent[0],boundary=(self.Global.lower,self.Global.upper))
            population = [np.vstack((population[0], self.Global.individual(offspring)[0])), np.vstack((population[1], self.Global.individual(offspring)[1]))]
            population, front_no, crowd_dis,_ = environment_selection(population, self.Global.N)
            self.eva = self.eva - self.Global.N
            if self.eva%(10*evaluation/self.ite) == 0:
                end = time.clock()
                print('Running time %10.2f, percentage %s'%(end-start,100*(evaluation-self.eva)/evaluation))
        print("=========================================")    
        return population

# Global = Global()

def solve(regress,Num):

    nsga = nsgaii()
    Global = Optset(Num)
    Global.load_regress(regress)
    nsga.setGlobal(Global)
    pop = nsga.run()
    optimal_policy = pop[0][-1].tolist()
    optimal_result = pop[1][-1].tolist()

    return optimal_policy,optimal_result

def main():
    ensemble = Ensemble()
    ensemble.load_model()
    optimal_policy,optimal_result = solve(ensemble, 2)
    print("optimal_policy : ", optimal_policy, " || optimal_result: ", optimal_result)


if __name__ == "__main__":

    main()
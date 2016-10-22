import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
import csv
import math
import sys

# Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR


#------------------------------------------------------------------------------
def getAValidrow(numOfFea, eps=0.015):
    sum = 0
    while (sum < 3):
       V = zeros(numOfFea)
       for j in range(numOfFea):
          r = random.uniform(0,1)
          if (r < eps):
             V[j] = 1
          else:
             V[j] = 0
       sum = V.sum()
    return V
#------------------------------------------------------------------------------

def Create_A_Population(numOfPop, numOfFea):
    population = random.random((numOfPop,numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population

def create_DE_population(numOfPop, numOfFea, fitness, old_pop):
    """Look for index of best fitness"""
    best_fitness_index = argmin(fitness)
    """Create empty Population"""
    new_pop = zeros((numOfPop, numOfFea))
    """Move best model(pop) from old to new pop"""
    new_pop[best_fitness_index] = old_pop[best_fitness_index]

    for index in range(0,numOfPop):
        new_vector = get_new_pop_vec(numOfPop,numOfFea,old_pop,index)
        #print str(shape(new_vector))
        #print str(shape(old_pop[index]))

        new_vector_fitness =cal_fitness_DE(new_vector)
        """Checks if is not the best fitness index for the given row,
            if is not equal compare new fitness vs old
            and move best row to new_pop"""
        if index != best_fitness_index:
            if(new_vector_fitness<fitness[index]):
                """Assign fitness"""
                fitness[index] = new_vector_fitness
                """Assign new Vector"""
                new_pop[index] = new_vector
            else:
                new_pop[index] = old_pop[index]

    return new_pop, fitness

"""creates the new vector from the three distinct vectors in the old population"""
def get_new_pop_vec(numOfPop, numOfFea, old_pop, index_position):
    """calculate new V, from 3 randomly selected distinct vectors not including the current row.
            F = 0.5
            V[i]= v[i]3 + F * ( v[i]2 - v[i]1)"""
    F = 0.5
    vi1, vi2, vi3 = create_three_random_v_indices(index_position, numOfPop)
    new_vector = [0]*numOfFea

    vector_1 = old_pop[vi1]
    vector_2 = old_pop[vi2]
    vector_3 = old_pop[vi3]

    for x in range(0,numOfFea):
        new_vector[x] = vector_3[x] + F * (vector_2[x] - vector_1[x])
    return new_vector

"""get fitness of the new vector"""

def cal_fitness_DE(new_vector):

    #print new_vector
    model = mlr.MLR()
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

    fittingStatus, fitness = FromFinessFileMLR.validate_single_model(model, new_vector, \
                                                              TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    return fitness


"""gathers the three indices needed to create the new child"""
def create_three_random_v_indices(current_vector_index, numOfPop):
    """Get vector index 1"""
    while True:
        v_index_1 = random.randint(0,numOfPop)
        if (v_index_1 != current_vector_index):
            break
    """Get vector index 2"""
    while True:
        v_index_2 = random.randint(0, numOfPop)
        if (v_index_2 != v_index_1 and v_index_2 != current_vector_index):
            break
    """Get vector index 3"""
    while True:
        v_index_3 = random.randint(0, numOfPop)
        if (v_index_3 != v_index_1 and v_index_3 != v_index_2 and v_index_3 != current_vector_index):
            break
    #print "v_index_1 = " + str(v_index_1) + "\nv_index_2 = " + str(v_index_2) + "\nv_index_3 = " + str(v_index_3) + "\ncurrent_vector_index = " + str(current_vector_index)
    return v_index_1, v_index_2, v_index_3




















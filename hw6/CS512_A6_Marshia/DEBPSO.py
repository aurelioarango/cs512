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
""" *Create Initial Velocity    ( V )
    Created randomly 50*396 - random number between 0 and 1"""
def create_initial_velocity(numOfPop, numOfFea ):

    initial_velocity = random.random((numOfPop, numOfFea))
    for i in range(0,numOfPop):
        for j in range(0, numOfFea ):
            initial_velocity[i][j]= random.uniform(0,1)

    return initial_velocity
#------------------------------------------------------------------------------
""" *getAValidrow function calculates 0.015 percent of the population """
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
""" *create the initial population - 50*396 - based on initial velocity"""
def create_initial_population(numOfPop, numOfFea, initial_velocity):

    population = zeros(numOfPop,numOfFea)
    lmbda = 0.015

    for i in range(numOfPop):
        for j in range(numOfFea):
            if initial_velocity[i][j] <= lmbda:
                population[i][j] = 1
            else:
                population[i][j] = 0
    return population
#------------------------------------------------------------------------------
"""*create new population- based on old population, velocity, global best, alpha and beta"""
def create_new_DEBPSO_population(numOfPop,numOfFea, alpha, population, velocity, globalbest, localbest):

    new_population = zeros(shape(population))
    beta = 0.004
    p = 0.5 * (1 + alpha)
    q = 1 - beta

    for i in range(0,numOfPop):
        for j in range(0,numOfFea):
            if alpha < velocity[i][j] and velocity[i][j] <= p:
                new_population[i][j] = localbest[i][j]
            elif p < velocity[i][j] and velocity[i][j] <= q:
                new_population[i][j] = globalbest[j]
            elif q < velocity[i][j]  and velocity[i][j] <= 1:
                new_population[i][j] = 1 - population[i][j]
            else:
                new_population[i][j] = population[i][j]

    """gets the new value for alpha for next iteration"""
    alpha = find_alpha(alpha, 20)

    return alpha, new_population

#------------------------------------------------------------------------------
""" *create new value of alpha. Alpha value decrese from 0.5 to 0.33,
    each time it will be reduced by 0.17/2000 (difference/number of iterations)."""
def find_alpha(previous_alpha,total_num_iterations ):
    new_alpha=previous_alpha - (.17/total_num_iterations)
    return new_alpha

#------------------------------------------------------------------------------
""" * update local best matrix """
def update_local_best_matrix(old_fitness, old_population, local_best_matrix, local_best_fitness, numOfPop, fileW):

    for i in range (numOfPop):
        if(old_fitness[i] < local_best_fitness[i]):
            local_best_matrix[i] = old_population[i]
    return local_best_matrix

#------------------------------------------------------------------------------
""" * update global best row """
def update_global_best(old_global_best_row,global_best_row_fitness, local_best_matrix, local_best_fitness):

    new_local_best_fitness_index = argmin(local_best_fitness)

    if(global_best_row_fitness > local_best_fitness[new_local_best_fitness_index]):
        old_global_best_row = local_best_matrix [new_local_best_fitness_index]
        global_best_row_fitness = local_best_fitness[new_local_best_fitness_index]

    return old_global_best_row, global_best_row_fitness

#------------------------------------------------------------------------------
""" *Update the velocity - using Deferential Evolution """
def update_velocity(numOfPop, numOfFea, population, initial_velocity):

    new_velocity = initial_velocity
    CR = 0.7

    for i in range (numOfPop):
        """get a new_vector based on 3 random vectors from old population"""
        new_vector = get_new_pop_vec(numOfPop, numOfFea, population, i)
        for j in range (numOfFea):
            """create a random number between 0 and 1 and check if it is less than CR """
            if (random.uniform(0,1) < CR):
                new_velocity[i][j] = new_vector[j]
            else:
                """ else, the velocity value remains the same """
                new_velocity[i][j] = initial_velocity[i][j]

    return new_velocity

#------------------------------------------------------------------------------
""" *creates the new_vector from 3 randomly selected distinct vectors from the old population not including the current row."""
""" using this formula:
        F = 0.5
        V[i]= v[i]3 + F * ( v[i]2 - v[i]1)"""
def get_new_pop_vec(numOfPop, numOfFea, old_pop, index_position):

    F = 0.5
    """call the function to get 3 random index position from 1 to 50"""
    vi1, vi2, vi3 = create_three_random_v_indices(index_position, numOfPop)
    new_vector = [0]*numOfFea

    vector_1 = old_pop[vi1]
    vector_2 = old_pop[vi2]
    vector_3 = old_pop[vi3]

    for x in range(0, numOfFea):
        new_vector[x] = vector_3[x] + F * (vector_2[x] - vector_1[x])

    return new_vector

#------------------------------------------------------------------------------
""" *randomly selects three distinct row index positions from 1 to 50 not including the current row index"""
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

#------------------------------------------------------------------------------
"""append to file"""
def append_to_file(new_vector,fileW):
    #print new_vector
    model = mlr.MLR()
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

    FromFinessFileMLR.validate_model_and_append(model,fileW, new_vector, \
                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

#------------------------------------------------------------------------------

















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


"""
Create Initial Velocity
Created randomly 50*385
How to find the initial velocity:
for (i=0; i<50; i++)
    for (j=0; j<385; j++)
    {
           V[i, j] = random number between 0 and 1; // this is not binary. It is between 0 and 1
    }
"""
def create_initial_velocity(numOfPop, numOfFea ):

    initial_velocity = random.random((numOfPop, numOfFea))
    for i in range(0,numOfPop):
        for j in range(0, numOfFea ):
            initial_velocity[i][j]= random.uniform(0,1)

    return initial_velocity
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
"""create new population- based on old population, velocity, global and local best"""
def create_new_BPSO_population(numOfPop,numOfFea, alpha, population, velocity, localbest, globalbest):

    new_population = zeros(shape(population))

    p = 0.5 * (1 + alpha)
    for i in range(0,numOfPop):
        for j in range(0,numOfFea):
            if velocity[i][j] <= alpha:
                new_population[i][j] = population[i][j]
            elif velocity[i][j] > alpha and velocity[i][j] <= p:
                new_population[i][j] = localbest[i][j]
            elif velocity[i][j] > p and velocity[i][j] <=1:
                new_population[i][j] = globalbest[j]
            else:
                new_population[i][j] = population[i][j]
    alpha = find_alpha(alpha, 2000)

    return alpha, new_population


def find_alpha(previous_alpha,total_num_iterations ):
    new_alpha=previous_alpha - (.17/total_num_iterations)
    return new_alpha

def update_local_best_matrix(old_fitness, old_population, local_best_matrix,local_best_fitness , numOfPop,fileW):

    for i in range (numOfPop):
        if(old_fitness[i] < local_best_fitness[i]):
            local_best_matrix[i] = old_population[i]
        """update each new vector"""
        append_to_file(local_best_matrix[i], fileW)
    return local_best_matrix

def update_global_best(old_global_best_row,global_best_row_fitness, local_best_matrix, local_best_fitness):

    new_local_best_fitness_index = argmin(local_best_fitness)

    if(global_best_row_fitness > local_best_fitness[new_local_best_fitness_index]):
        old_global_best_row = local_best_matrix [new_local_best_fitness_index]
        global_best_row_fitness = local_best_fitness[new_local_best_fitness_index]

    return old_global_best_row, global_best_row_fitness


"""    Update the velocity """
def update_velocity(numOfPop,numOfFea,old_velocity,population,local_best_matrix, global_best_row):
    c1=2
    c2=2
    inertiaWeight =0.9
    new_velocity = zeros(shape(old_velocity))
    for i in range (numOfPop):
        for j in range (numOfFea):
            term1 = c1 * random.random() * (local_best_matrix[i][j] - population[i][j])
            term2 = c2 * random.random() * (global_best_row[j] - population[i][j])
            new_velocity[i][j] = (inertiaWeight * old_velocity[i][j]) + term1 + term2

    return new_velocity

def Create_A_Population(numOfPop, numOfFea):
    population = random.random((numOfPop,numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        """ getAValidrow function calculates 0.015 percent of the population """
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population


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

    for x in range(0, numOfFea):
        new_vector[x] = vector_3[x] + F * (vector_2[x] - vector_1[x])
    return new_vector
"""get fitness of the new vector"""
def append_to_file(new_vector,fileW):
    #print new_vector
    model = mlr.MLR()
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

    FromFinessFileMLR.validate_model_and_append(model,fileW, new_vector, \
                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

"""get fitness of the new vector"""
def cal_fitness_DE(new_vector):
    #print new_vector
    model = mlr.MLR()
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

    fitness = FromFinessFileMLR.validate_single_model(model, new_vector, \
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




















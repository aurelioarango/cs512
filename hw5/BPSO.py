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
def create_new_BPSO_population( ):
    """"""

def find_alpha(previous_alpha,total_num_iterations ):
    new_alpha=previous_alpha - (.17/total_num_iterations)

    return new_alpha

def Create_A_Population(numOfPop, numOfFea):
    population = random.random((numOfPop,numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population

def create_DE_population(numOfPop, numOfFea, fitness, old_pop,fileW):
    """Look for index of best fitness"""
    best_fitness_index = argmin(fitness)
    """Create empty Population"""
    new_pop = zeros((numOfPop, numOfFea))
    """Move best model(pop) from old to new pop"""
    new_pop[best_fitness_index] = old_pop[best_fitness_index]
    new_fitness = zeros(numOfPop)

    for index in range(0,numOfPop):
        new_vector = get_new_pop_vec(numOfPop,numOfFea,old_pop,index)
        #print str(shape(new_vector))
        #print str(shape(old_pop[index]))

        new_vector_fitness = cal_fitness_DE(new_vector)

        """Checks if is not the best fitness index for the given row,
            if is not equal compare new fitness vs old
            and move best row to new_pop"""
        if (index != best_fitness_index):


            if(new_vector_fitness < fitness[index]):
                #print str(new_vector_fitness) + " is new vector fitness *******"
                """Assign fitness"""
                new_fitness[index] = new_vector_fitness
                """Assign new Vector"""
                new_pop[index] = new_vector
                """Appending to file only new vectors"""
                append_to_file(new_vector,fileW)
                #print str(new_fitness[index]) + " value stored in new fitness\n"
            else:
                #print str(fitness[index]) + " is old vector fitness"
                new_pop[index] = old_pop[index]
                new_fitness[index] = fitness[index]

        else:
            new_fitness[index] = fitness[index]
    return new_pop, new_fitness

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




















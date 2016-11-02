""" HW5 (BPSO)
        Binary Particle swarm optimization
    CS512
    Aurelio Arango, Kristina Nystrom, Marshia Hashemi  """
"""
1)
    Make initial population 50x385 - randomly
2)
    Create Initial Velocity
        Created randomly 50*385
        How to find the initial velocity:
        for (i=0; i<50; i++)
            for (j=0; j<385; j++)
            {
                   V[i, j] = random number between 0 and 1; // this is not binary. It is between 0 and 1
            }
3)
    Initialize Local Best Matrix (same as initial population)
4)
    Create Global best row
        the row with the best fitness
5)
    Find the fitness of each row

6)
   Find the new population
    -	Find the value of alpha
        -   During the 2000 iterations, the value of alpha ranges from 0.5 to 0.33.
            So the difference between 0.5 and 0.33 is (0.5 - 0.33 = 0.17). Thus, in order to
            reduce the value of alpha in each iteration (2000 iterations) we need to divide
            0.17 by 2000 to know how much in each iteration we need to subtract from the
            value of alpha
7)
    Update the local best fitness
        Updating the new local best Matrix:
        For each row "i" of the current population
        If the fitness of the population[i] < fitness of local-best[i]
        Local-best[i] = population[i]
8)
    Update the global best row
        Global-best row = the row of the local-matrix with the lowest fitness
9)
    Update the velocity
        Find the Velocity matrix as follows: (c1=2, c2=2 by default, inertiaWeight = 0.9)
        term1 = c1*random.random()*(localBestMatrix[i][j]-population[i][j])
        term2 = c2*random.random()*(globalBestRow[j]-population[i][j])
        velocity[i][j]=(inertiaWeight*velocity[i][j])+term1+term2

    Notes:
    Alpha value starts from .5 and stopping is .33,
    How much should be reduce by .17/2000 (difference/number of iterations).

    Beta 0.004
 """


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
import BPSO

"""Create the output file"""

fileW = FromFinessFileMLR.createAnOutputFile()
#fileW = 0
model = mlr.MLR()

#Number of descriptor should be 396 and number of population should be 50 or more
"""Number of population"""
numOfPop = 50
"""Number of total features"""
numOfFea = 385

# Final model requirements

R2req_train    = .6
R2req_validate = .5
R2req_test     = .5

TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

""" Generate randomly population with 385 features and 50 pop """

population = BPSO.Create_A_Population(numOfPop, numOfFea)
""" Get fitness"""
fitness = FromFinessFileMLR.validate_model(model, fileW, population, \
                                                TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
"""Initialize velocity"""
initial_velocity=BPSO.create_initial_velocity(numOfPop,numOfFea)
print initial_velocity
#print str(shape(initial_velocity))

"""Initialize Local Best Matrix (Same as Initial Population)"""
local_best_matrix = population

"""Create Global best row"""
global_best_row_index = argmin(fitness)
global_best_row=population[global_best_row_index]
#global_best_row=population[argmin(fitness)]


""" HW6 (DE-BPSO)
        Deferential Evolution - Binary Particle swarm optimization
    CS512
    Marshia Hashemi  """
"""
******************
*1)  V = Create Initial Velocity
        Created randomly 50*396
        for (i=0; i<50; i++)
            for (j=0; j<396; j++)
            {
                   V[i, j] = random number between 0 and 1; // this is not binary. It is between 0 and 1
            }
*2)  X = Create initial population 50*396
        for (i=0; i<50; i++)
            for (j=0; j<396; j++)
            {
                   if (V[i,j] <= Lambda)  // The value of Lambda is 0.01
                        X[i,j] = 1
                    else
                        X[i,j] = 0
            }
*3)  calculate the fitness of the population
*4)  P = Initialize Local Best Matrix (same as initial population)
    P = X

    G = Create Global best row
        the row with the best fitness

******************
5)  Update the velocity - Using DE
        Find the Velocity matrix as follows:
	for (i=0; i<50; i++)
		for (j=0; j<396; j++)
		{
			Randomly select 3 rows from the populations and call them as r1, r2, and r3
			Let r = r3 + F * (r2 - r1) // the value of F should be set to 0.7
			// Do the cross mutation of row "i" and "r"
			if ((random between 0 and 1) < CR) // not binary, CR = 0.7
				V[i,j] = r[j]
			else
				V[i,j] = V[i,j] // remains unchanged
		}

6)  Find the new population
    -	Find the value of alpha
        Alpha value starts from 0.5 and is decremented to 0.33,
        How much should be reduce by 0.17/2000 (difference/number of iterations).
        Beta = 0.004

	for (i=0; i<50; i++)
		for (j=0; j<396; j++)
		{
			if ( (alpha < V[i,j]) && (V[i,j] <= 0.5*(1+alpha))
				X[i, j] = P[i,j];
			else if (  (0.5*(1+alpha)) < V[i,j]) && (V[i,j] <= (1-beta))
				X[i,j] = G[j] // the global vector value
			else if  (1-beta) < V[i,j]) && (V[i,j] <=1))
				X[i.j] = 1 - X[i,j]
			else
				X[i,j] = X[i,j]; // remains unchanged
		}
7)  Find the fitness of each row
8)  Update the local best fitness
        Updating the new local best Matrix:
        For each row "i" of the current population
        If the fitness of the population[i] < fitness of local-best[i]
        Local-best[i] = population[i]

9)  Update the global best row
        Global-best row = the row of the local-matrix with the lowest fitness
******************
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
import DEBPSO

"""Create the output file"""

fileW = FromFinessFileMLR.createAnOutputFile()
#fileW = 0
model = mlr.MLR()
start = time.time()
#Number of descriptor should be 396 and number of population should be 50 or more
"""Number of population"""
numOfPop = 50
"""Number of total features"""
numOfFea = 396

# Final model requirements

R2req_train    = .6
R2req_validate = .5
R2req_test     = .5
alpha = 0.5


TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

#unfit = 1000
#fittingStatus = unfit

"""Initialize velocity"""
initial_velocity = DEBPSO.create_initial_velocity(numOfPop, numOfFea)

""" Initialize population"""
population = DEBPSO.create_initial_population(numOfPop, numOfFea, initial_velocity)

""" Get fitness of the initial population"""
fitness = FromFinessFileMLR.validate_model(model, fileW, population, \
                                                TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

"""Initialize Local Best Matrix (Same as Initial Population)"""
local_best_matrix = population

"""Initialize Global best row"""
global_best_row_index = argmin(fitness)
global_best_row_fitness = fitness[global_best_row_index]
global_best_row=population[global_best_row_index]


"""number of generation we want the program to run"""
# change it to 2000
generations_to_run = 20

print "Starting program"


"""run the program for number of generations - 2000"""
for i in range (0, generations_to_run):


    """ call new population- based on old population, velocity, global and local best, alpha and beta"""
    alpha, population = DEBPSO.create_new_DEBPSO_population(numOfPop, numOfFea, alpha, population, initial_velocity, global_best_row, local_best_matrix)

    """"Calculate local best fitness of the new population"""
    local_best_matrix_fitness = FromFinessFileMLR.validate_model(model, fileW, population, \
                                                TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    """Update local best matrix"""
    local_best_matrix = DEBPSO.update_local_best_matrix(fitness, population, local_best_matrix, local_best_matrix_fitness, numOfPop, fileW)

    """update global best row"""
    global_best_row, global_best_row_fitness = DEBPSO.update_global_best(global_best_row, global_best_row_fitness, local_best_matrix, local_best_matrix_fitness)

    """ update velocity """
    initial_velocity = DEBPSO.update_velocity(numOfPop, numOfFea, population, initial_velocity)


end = time.time()
print str((end - start) / 60) + " minutes"

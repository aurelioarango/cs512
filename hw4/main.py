""" HW4
    CS512
    Aurelio Arango, Kristina Nystrom, Marshia Hashemi  """
"""
1) Generate population with 385 features and 50 pop
2) get fitness for the population
3) grab best model-fitness (row) and move it to new pop in same index
4) calculate new V, from 3 randomly selected distinct vectors not including the current row.
    F = 0.5
    V[i]= v[i]3 + F * ( v[i]2 - v[i]1)
5) calculate its fitness and compare to the old vector.
    if the new fitness is better than the old one then replace it in the pop
     otherwise, keep old vector.

Compute a better fitness function """

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
import DifferentialEvolution

fileW = FromFinessFileMLR.createAnOutputFile()
#fileW = 0
model = mlr.MLR()

#Number of descriptor should be 396 and number of population should be 50 or more
"""Number of population"""
numOfPop = 50
"""Number of total feautres"""
#numOfFea = 396
numOfFea = 385

# Final model requirements

R2req_train    = .6
R2req_validate = .5
R2req_test     = .5

TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

unfit = 1000
fittingStatus = unfit
""" Generate population with 385 features and 50 pop"""

population = DifferentialEvolution.Create_A_Population(numOfPop, numOfFea)
""" Get fitness for the population"""
fittingStatus, fitness = FromFinessFileMLR.validate_model(model, fileW, population, \
                                                TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

"""Create functions in the Differential Evolution class for the following: """
"""grab best model-fitness (row) and move it to new pop in same index"""
#DifferentialEvolution.create_DE_population(numOfPop, numOfFea,fitness,population)
generations_to_run = 2
fit = fitness
iterations_since_best_fitness_has_changed = 0

for i in range(0,generations_to_run):
    print "Running Generation " + str(i)
    if iterations_since_best_fitness_has_changed == 500:
        print "Program stopped - the best fitness has not changed in 500 generations"
        break

    fitness = fit
    best_fitness_index = argmin(fitness)
    pop, fit = DifferentialEvolution.create_DE_population(numOfPop, numOfFea,fitness,population,fileW)
    """Appending whole generations, can potentially include duplicates"""
    #FromFinessFileMLR.validate_model(model, fileW, pop, \
    #                                            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
    if argmin(fit)==argmin(fitness):
        iterations_since_best_fitness_has_changed = iterations_since_best_fitness_has_changed + 1
#print str(shape(pop))
#print fitness
#print fit
#print "\n"

#for x in range(0, numOfPop):
#    if(fitness[x] != fit[x]):
#        print "x = " + str(x) + " : (fitness) " + str(fitness[x]) + " != (fit) " + str(fit[x])


"""calculate its fitness and compare to the old vector.
    if the new fitness is better than the old one then replace it in the pop
     otherwise, keep old vector."""

#print fitness

#print len(fitness)

#print str(argmin(fitness)) +" "+str( fitness[argmin(fitness)])

#DifferentialEvolution.create_DE_population(numOfPop, numOfFea,fitness,population)

"""5 is the index position we are currently trying to fill"""
#DifferentialEvolution.create_three_random_v_indices(5, numOfPop)

#fit = DifferentialEvolution.cal_fitness_DE(population[0])
#print fit


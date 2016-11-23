"""
  Aurelio Arango
  HW6
  DE-BPSO
  CS512
"""
"""

1) Create the initial velocity, Initialize velocity to random real value between 0 to 1

  for i in range(0,50):
    for j in range (0, 396):
      velocity[i][j] = random.random(0,1)

2) Create the initial Population

(same as before)

3) find the fitness of the population

(same as before)

4) Initialize local-best matrix

local_best_matrix = population;

5) Initialize global-best row
  find best fitness index and get its row

Repeat:

6) Find the new velocity

for i in range (0,50):
  for j in range (0,396):
    # get three random rows from pop r1, r2, r3
    F = 0.7
    CR = 0.7
    # Let row = r3 + F * (r2 - r1)
       #Cross mutation row i and r
       if(random.random(0,1) < CR)
        velocity[i][j] = r[j]
       else
        do nothing
7) Create new population
for i in range(0, 50):
    for j in range(0, 396):
        if alpha < velocity[i][j] and velocity[i][j] <= .5 * (1 + alpha):
            population[i][j] = local_best_matrix[i][j]
        elif 0.5 * (1 + alpha) < velocity[i][j] and velocity[i][j] <= (1 - beta):
            population[i][j] = global_best_row[j]
        elif 1 - beta < velocity[i][j] and velocity[i][j] <= 1
            population[i][j] = 1 - population[i][j]
        else
            keep old population

8) calculate the fitness of the new population

9) update the local best matrix (sames as hw5)

10) Update global best row, if necessary

Repeat

11) Stop loop if no changes after 500 iterations otherwise, continue until alpha reaches its limit

"""

import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
import csv
import math
import sys

# Local files created by me
import DE_BPSO_model
import FromDataFileMLR
import FromFinessFileMLR
import DE_BPSO

"""Create the output file"""

print "Starting program"

fileW = FromFinessFileMLR.createAnOutputFile()
#fileW = 0
model = DE_BPSO_model.DE_BPSO_MODEL()
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
beta = 0.004

TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

population = DE_BPSO.Create_A_Population(numOfPop, numOfFea)
""" Get fitness"""
fitness = FromFinessFileMLR.validate_model(model, fileW, population, \
                                                TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
"""Initialize velocity"""
velocity = DE_BPSO.create_initial_velocity(numOfPop,numOfFea)

"""Initialize Local Best Matrix (Same as Initial Population)"""
local_best_matrix = population


""""Calculate local best fitness of the local_best_matrix"""
local_best_matrix_fitness = FromFinessFileMLR.validate_model(model, fileW, local_best_matrix, \
                                                             TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

"""Create Global best row"""
global_best_row_index = argmin(fitness)
global_best_row_fitness = fitness[global_best_row_index]
global_best_row=population[global_best_row_index]



generations_to_run = 2000
fitness_change = global_best_row_fitness
last_change =0

"""<Insert for loop>"""
for i in range (0, generations_to_run):

    if fitness_change == 500:
        print "Stopping program, best fitness has not change in 500 generations."
        break

    """update velocity with population"""
    velocity = DE_BPSO.update_velocity(numOfPop, numOfFea, velocity,population)


    """update-population / create-new-population , with velocity"""

    population = DE_BPSO.update_population_DE_BPSO(numOfPop,numOfFea,velocity,alpha,generations_to_run, \
                                                       population,local_best_matrix,global_best_row)
    """calculate fitness of new population"""

    new_fitness = FromFinessFileMLR.validate_model(model, fileW, population, \
                                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)


    """update local best _matrix"""
    local_best_matrix = DE_BPSO.update_local_best_matrix(fitness,population,local_best_matrix, \
                                                         local_best_matrix_fitness,numOfPop,fileW)


    """"update local best matrix fitness """
    local_best_matrix_fitness = FromFinessFileMLR.validate_model(model, fileW, local_best_matrix, \
                                                                 TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    """ update global best row"""
    global_best_row, global_best_row_fitness = DE_BPSO.update_global_best(global_best_row,global_best_row_fitness, \
                                                                          local_best_matrix,local_best_matrix_fitness)

    """increment if the fitness has not change, else update last_change and set fitness change"""
    if fitness_change  == global_best_row_fitness:
        last_change += 1;
    else:
        last_change = 0
        """update the last time fitness change"""
        fitness_change = global_best_row_fitness

end = time.time()
print str((end - start) / 60) + " minutes"



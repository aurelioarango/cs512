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
    """calculate new V, from 3 randomly selected distinct vectors not including the current row.
        F = 0.5
        V[i]= v[i]3 + F * ( v[i]2 - v[i]1)"""


def get_new_pop(numOfPop, numOfFea, fitness, old_pop):
    """calculate new V, from 3 randomly selected distinct vectors not including the current row.
            F = 0.5
            V[i]= v[i]3 + F * ( v[i]2 - v[i]1)"""

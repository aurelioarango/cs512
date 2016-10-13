"""CS512 Quick Sort Algorithm
   Team Members: Aurelio Arango

   This algorithm sorts a GA population by its fitness.
 """

class Sort:

    def __init__(self):
        """Initialization"""
        self.coef = None

def quick_sort_population(population, fitness, iLow, iHigh):
    pivotpoint = 0
    """print str(len(fitness)) +" quick"
    if len(fitness) == 2:
        #swap values
        if fitness[0] > fitness[1]:
            temp = fitness[1]
            fitness[1] = fitness[0]
            fitness[0]=temp
    elif len(fitness) > 2:"""
    # print len(fitness)
    if iHigh > iLow:
        pivotpoint = sort_partition(population, fitness, iLow, iHigh)
        quick_sort_population(population, fitness, iLow, pivotpoint)
        quick_sort_population(population, fitness, pivotpoint + 1, iHigh)

def sort_partition(population, fitness, iLow, iHigh):
    """Partition of quick sort """
    # i=iLow+1
    j = iLow
    temp = 0
    pivot_item = fitness[iLow]

    for i in range(iLow + 1, iHigh):
        """"""
        if (fitness[i] < pivot_item):
            j = j + 1
            temp = fitness[i]
            fitness[i] = fitness[j]
            fitness[j] = temp
            temp_pop = population[i]
            population[i] = population[j]
            population[j] = temp_pop
    pivot_point = j
    temp = fitness[iLow]
    fitness[iLow] = fitness[pivot_point]
    fitness[pivot_point] = temp
    temp_pop = population[iLow]
    population[iLow] = population[pivot_point]
    population[pivot_point] = temp_pop

    return pivot_point

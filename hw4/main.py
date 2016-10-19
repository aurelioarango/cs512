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

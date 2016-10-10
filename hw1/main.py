#!/usr/bin/python
import MyMatrix
import sys

""" Team 1, CS512
Aurelio Arango
Kristina Nystrom
Marshia Hashemi

"""
""" Write an object oriented program in python that does the following:

Part a)
- Interactively get a positive number, n, from a user
    If n is less or equal to 3 report an error and quit the program.
-Otherwise, try to get n*n numbers from the first file.
    For example, if n=5 you are required to get 25 numbers from the first file in order to make a 5 by 5 matrix.
Therefore,
    -If there is less than 25 numbers in the file, report error and quit the program
    -If there are more than 25 numbers (for size 5) from the file, get the first 25 numbers, make
 a matrix of 5 by 5 and ignore the other numbers

Part b)
-Try to get n*n numbers from the second file.
    If the value of n to create the first matrix, you must use the same value n to create the second matrix.
-The two matrix should have the same size. So if the first size of the first matrix is 5, the size of the second
    matrix must be 5 too.
-Again read the integer number from the second file.
    For n=5, If there is less than 25 numbers (for size = 5) in the second file, report error and quit the program
    If there are more than 25 numbers from the file, get the first 25 numbers, make the second matrix of 5 by 5 and
    ignore the other numbers

Part c)
-Do the product of the two matrices and print the resulting matrix with proper heading
-Do the dot product of the two matrices and print the resulting matrix with proper heading
-Transpose the first and the second matrices and multiply them with each other and print the resulting matrix
    with proper heading.
-Do the dot product of the transposed matrices and print the resulting matrix with proper heading
-Divide the first matrix by the second one. In case of division by zero you need to show the word undefined
    in the proper matrix location. Print the resulting matrix with proper heading
 """

"""" Part a)"""
# instantiate matrix object
my_matrix = MyMatrix.MyMatrix()

# Get matrix size
matrix_size = my_matrix.get_size()

# Load matrix
m1 = my_matrix.get_matrix(matrix_size, "file1.txt")

""" Part b)"""
# Load matrix
m2 = my_matrix.get_matrix(matrix_size, "file2.txt")
""" Part C"""

"""  Get product of m1 and m2"""
m1_multiply_m2 = my_matrix.product(m1, m2)

""" Get dot product of m1 and m2 """
m1_dot_multiply_m2 = my_matrix.dot_product(m1, m2)

""" transpose m1 """
m1_trans = my_matrix.transpose(m1)

""" transpose m2 """
m2_trans = my_matrix.transpose(m2)

""" Get product of m1_trans and m2_trans """
m1_trans_multiply_m2_trans = my_matrix.product(m1_trans, m2_trans)

""" Get dot product of m1_trans and m2_trans """
m1_trans_dot_multiply_m2_trans = my_matrix.dot_product(m1_trans, m2_trans)

"""" m1_divided_m2 """
m1_divided_m2 = my_matrix.divide(m1, m2)

""" Printing Matrices: """
""" Printing m1 """
my_matrix.print_matrix(m1, "\nThe content of the first matrix is:\n")

""" Printing m2: """
my_matrix.print_matrix(m2, "\nThe content of the second matrix is:\n")

""" Printing multiplication if m1 and m2: """
my_matrix.print_matrix(m1_multiply_m2, "\nThe product of the two matrices is:\n")

""" Printing dot multiplication of m1 and m2: """
my_matrix.print_matrix(m1_dot_multiply_m2, "\nThe dot-product of the two matrices is:\n")

""" Printing matrix m1 divided matrix m2"""
my_matrix.print_matrix(m1_divided_m2, "\nThe result of matrix1 divided by matrix2 is:\n")

""" Printing transposed matrix of m1: """
my_matrix.print_matrix(m1_trans, "\nThe transpose of the first matrix is:\n")

""" Printing transposed matrix of m2 """
my_matrix.print_matrix(m2_trans, "\nThe transpose of the second matrix is:\n")

""" Printing multiplication of transposed matrix m1 and transposed matrix m2"""
my_matrix.print_matrix(m1_trans_multiply_m2_trans, "\nThe product of the transpose of the two matrices is:\n")

""" Printing dot multiplication of transposed matrix m1 and transposed matrix m2"""
my_matrix.print_matrix(m1_trans_dot_multiply_m2_trans, "\nThe dot-product of the transpose of the two matrices is:\n")

""" Printing end of program message """
my_matrix.print_end_of_program()

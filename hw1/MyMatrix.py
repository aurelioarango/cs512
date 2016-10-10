#!/usr/bin/python
"""Created By Aurelio Arango, Marshia Hashemi, & Kristina Nystrom
    Log Change:
    -Sep 5, 2016 Initial File Created
    -Sep 2-2016, Added dot product function
    -sep 6-2016, Added matrix division function
    -sep 13-2016, Made get_file(function) with exception handling
                  change int to floats,
                  added <= to check for values less than 3
    -sep 14-2016, added < to check for values greater than 6
                  change delimiter to \t (tab)
                  change self.logger.write(to to \b (tab) delimited
"""
# Matrix Class that gets file data,
import sys
import pprint
from Logger import Logger

class MyMatrix:

    def __init__(self):
        self.n_size = 0  # Set size of the matrix
        self.logger = Logger() # Creates a logger for command screen

    def get_size(self):
        size = raw_input("\n>> Enter the dimension of your matrix: ")
        user_input = size # makes a copy of the user input to output the value in case of error
        """Converts user's input to an integer"""
        try:
            size = int(size)
        except:
            self.logger.write("\nError: ***** This dimension ("+user_input+") is out of bound. The program stops in here.\n")
            sys.exit(1)
        self.n_size = size

        """Checks to see if user entered a number greater than 3"""
        if(size <= 3):
            self.logger.write("\nError: ***** This dimension ("+user_input+") is out of bound. The program stops in here.\n")
            sys.exit(1)

        #elif(size>6):
            #self.logger.write("\nError: ***** ("+user_input+") We can only handle up to 6 dimension at this time. The program stops in here.\n")
            #sys.exit(1)
        else:
            return size

    def get_file(self,file_name):
        try:
            """Open file"""
            opened_file = open(file_name)
        except:
            self.logger.write("Error: file "+file_name+" could not be found\n")
            sys.exit(1)
        return opened_file

    def get_matrix(self, n_size, file_name):
        """ Initialize Matrix n x n with 0"""
        Matrix = [[0 for x in range(n_size)] for y in range(n_size)]
        """ Set total expected elements for matrix"""
        total_nums_to_read = n_size * n_size
        """ Counter for the number of values in the file"""
        counter_nums = 0
        """Try to get file"""
        opened_file = self.get_file(file_name)
        x = 0
        y = 0
        for line in opened_file:
            # Split a string into tokens separated by a space
            y = 0
           # self.logger.write(line)  #Test line items
            if x < n_size:
                for string_nums in line.split('\t'):
                    """ Check if the have elements is less than the numbers to be read"""
                    if y < n_size :
                        """ Convert numbers from characters to integers"""
                        if string_nums != " ":
                            #self.logger.write(string_nums)
                            num = int(float(string_nums))
                            #num = string_nums
                        """ Add the new number to the matrix"""
                        Matrix[x][y] = num
                        #self.logger.write("Num:"+str(num)+" C:"+str(counter_nums))
                        """"Increment total count"""
                        counter_nums += 1
                        """Move to the next column"""
                        #self.logger.write(Matrix)
                        y += 1
                """Move to the next row"""
                #self.logger.write("\n")
                x += 1
            #self.logger.write(Matrix)
        #  Close file

        opened_file.close()
        """Check if enough values were read in the file"""
        if counter_nums < total_nums_to_read:
            sys.exit("Not enough values in the file: \n")
        else:
            return Matrix

    def product(self, first_matrix, second_matrix):
        """Initializing the computed matrix of size n by n to 0"""
        cmatrix = [[0 for x in range(self.n_size)] for y in range(self.n_size)]
        for i in range(0, self.n_size):
            for j in range(0, self.n_size):
                cmatrix[i][j] = 0
                """ Multiply the first_matrix row by the second_matrix column to get the value
                    of the computed matrix"""
                for k in range(0, self.n_size):
                    cmatrix[i][j] = cmatrix[i][j] + first_matrix[i][k] * second_matrix[k][j]
        return cmatrix

    def dot_product(self, first_matrix, second_matrix):
        """Dot product of a Matrix is the same as the product of a Matrix"""
        """ Dot product of two vectors is the summation of the inner product. A1B1 + A2B2 +A3B3...AnBn"""

        dmatrix = [[0 for x in range(self.n_size)] for y in range(self.n_size)]
        for i in range(0, self.n_size):
            for j in range(0, self.n_size):
                dmatrix[i][j] = first_matrix[i][j] * second_matrix[i][j]
        return dmatrix

    def transpose(self, first_matrix):
        """Create computed matrix n x n to hold values of the computed matrix"""
        tmatrix = [[0 for x in range(self.n_size)] for y in range (self.n_size)]
        for i in range(0, self.n_size):
            """ Exchange the row of a matrix with the column of the other"""
            for j in range(0, self.n_size):
                tmatrix[i][j] = first_matrix[j][i]
        return tmatrix

    def divide(self, first_matrix, second_matrix):
        dmatrix = [[" " for x in range(self.n_size)] for y in range(self.n_size)]
        for i in range (0, self.n_size ):
            for j in range (0, self.n_size):
                if second_matrix[i][j] == 0:
                    dmatrix[i][j] = "undefined"
                else:
                    dmatrix[i][j] = float("{0:2f}".format( float(first_matrix[i][j]) / float(second_matrix[i][j])))
        return dmatrix

    def print_matrix(self, matrix_print, message):
        """ Get matrix matrix to self.logger.write("""
        self.logger.write(message)
        for i in range(0, self.n_size):
            for j in range(0,self.n_size):
                # Trailing , to ignore newline
                self.logger.write("\t"+str(matrix_print[i][j]),)
            #self.logger.write((matrix_print[i][:]))
            self.logger.write("\n")
        self.logger.write("====================================================================================\n")

    def print_end_of_program(self):
        self.logger.write("\n\n(************************************End of the Program *******************************)\n")

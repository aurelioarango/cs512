#!/usr/bin/python

""" Team 1, CS512
Aurelio Arango
Kristina Nystrom
Marshia Hashemi

"""

import sys

#Class that prints output to file as well as terminal
#writes to file logfile.log
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        ''' opens logfile.log with append properties '''
        self.log = open("logfile.log", "a")

    ''' writes to both the log file and to the terminal '''
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

sys.stdout = Logger()
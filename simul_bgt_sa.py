#!/usr/bin/env python

# ********* #
# Libraries #
# ********* #

import os
import numpy as np
import pickle
import csv

from source.bgt_network import *
from source.network import *
from source.attractors import *
from optim.simulated_annealing import *

# ********** #
# Parameters #
# ********** #

# seed
np.random.seed(92)

# Parameters
input_length = 1000
trigger_length = 20
nb_triggers = 20

temperature = 10.0    # Initial temperature
cooling_rate = 0.99   # Cooling rate (close to 1 means slower cooling)


# ************** #
# Initialization #
# ************** #

# BGT Network
# network = generate_network(nb_inputs=2, nb_nodes=8, nb_input_connections=4, nb_internal_connections=16)
# network_M = net_to_matrix(network)
network = (input_nodes_N, nodes_N, edges_N)
network_M = net_to_matrix(network)

# Apply random distortion to weight matrix M[0]
# distort(M[0], noise=noise)

# Input
U, ticks = generate_input(input_dim=network_M[1].shape[0], 
                          input_length=input_length, 
                          mode="poisson", 
                          lamda=3, 
                          triggers=True, 
                          nb_triggers=nb_triggers, 
                          trigger_length=trigger_length)

print(ticks)


# **** #
# MAIN #
# **** #

if __name__ == "__main__":

    M, best_energy, temperature = simulated_annealing(initial_solution=network_M, 
                                                      func=attractor_energy, 
                                                      max_iterations=input_length, 
                                                      initial_temp=temperature, 
                                                      cooling_rate=cooling_rate)
#!/usr/bin/env python

# ********* #
# Libraries #
# ********* #

import os
import numpy as np
import pickle
import csv
from tqdm import tqdm

from source.bgt_network import *
from source.network import *
from source.attractors import *

# ********** #
# Parameters #
# ********** #

np.random.seed(13)
verbose = False
eta_in = 0.15
plumb = 1
bounds = (-0.1999, 0.5999)
noise = (-0.1, 0.5)
input_size = 3000 # Adjust as necessary
memory, memory_length = 149, 1
input_dim = 1
nb_triggers = 26
trigger_length = 10

cwd = os.getcwd()
output_file = os.path.join(cwd, "runs", f"new_simul_{nb_triggers}.csv")


# ***************************** #
# Initialization of Simulations #
# ***************************** #

def initialize_simulation():
    """Initializes the simulation components: network, matrix, weight distortions, and attractors."""

    # Generate BGT network and matrix
    N = (input_nodes_N, nodes_N, edges_N)
    M = net_to_matrix(N)

    # Apply random distortion to weight matrix M[0]
    distort(M[0], noise=noise)

    # Prepare a non-modifiable copy for the STDP rule
    A_default = np.copy(M[0])

    # Create network and automaton from the distorted matrix
    A_new = netM_to_aut(M)

    # Compute attractors
    dico_cycles = get_simple_cycles(A_new)
    SCC = largest_list(list(dico_cycles.keys()))
    _, n = dico_cycles[SCC]

    # Initialize results storage
    cycles_list = [n]

    # Write initial simulation parameters to the output file
    with open(output_file, "w") as f:
        f.write("memory_length,nb_attractors,min_attractors,max_attractors,eta,tick\n")
        f.write(f"{memory_length}, {n}, {n}, {n}, {eta_in}, 0\n")

    return M, A_default, cycles_list


# ********** #
# Simulation #
# ********** #

def run_simulation(U, ticks, M, A_default, cycles_list):
    """
    Runs the main simulation loop, updating network states, computing cycles, and logging results.

    Args:
        U (dict): Input stream with trigger patterns.
        ticks (list): Positions where trigger patterns end.
        M (list): Current network matrices and current state.
        A_default (np.ndarray): Original weight matrix for the STDP rule.
        cycles_list (list): List to store cycle counts for tracking.
    """
    eta = eta_in

    for nb_iter in tqdm(range(len(U) - 2)):
        
        # update tick
        tick = 0
        if nb_iter in ticks:
            tick = 1

        # Update memory_length
        global memory_length
        memory_length = memory_length + memory if nb_iter in ticks else max(1, memory_length - 1)

        # Extract next input segment
        current_input = {0: U[nb_iter]} # , 1: U[nb_iter + 1]} # XXX

        # Simulate network behavior and apply STDP rule
        x = M[-1] # current state
        _, _, M = simulation(M[0], M[1], M[2], M[3], x, 
                                    current_input,
                                    epoch=len(current_input), 
                                    stdp=[A_default, eta, plumb, bounds])
        x_plus = M[-1].reshape(-1, 1) # new state

        # Compute cycles for the current network state
        A_new = netM_to_aut(M)
        dico_cycles = get_simple_cycles(A_new)
        SCC = largest_list(list(dico_cycles.keys()))
        _, n = dico_cycles[SCC]

        # Update cycle tracking and STDP parameters
        stack_operation(cycles_list, memory_length, n)
        min_cycles, max_cycles = min(cycles_list), max(cycles_list)
        eta = linear(min_cycles, eta_in, max_cycles, eta_in / 20.0, n)

        # Logging
        if verbose:
            print(current_input, M[4], M[0], M[1], cycles_list, eta)

        # Write results to file
        with open(output_file, "a") as f:
            f.write(f"{memory_length}, {n}, {min_cycles}, {max_cycles}, {eta}, {tick}\n")


if __name__ == "__main__":
    # Initialize simulation
    M, A_default, cycles_list = initialize_simulation()
    
    # Generate input and run simulation
    U, ticks = generate_input(input_dim=input_dim, input_size=input_size, mode="random", lamda=5, 
                              triggers=True, nb_triggers=nb_triggers, trigger_length=trigger_length)
    
    run_simulation(U, ticks, M, A_default, cycles_list)

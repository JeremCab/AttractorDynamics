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

np.random.seed(95)
verbose = False
eta_in = 0.15
plumb = 1
bounds = (-0.1999, 0.5999)
noise = (-0.1, 0.5)
input_dim = 1
input_size = 3000 # Adjust as necessary
memory, memory_length = 149, 1
nb_triggers = 19
trigger_length = 10

cwd = os.getcwd()
output_file = os.path.join(cwd, "runs", f"GSP_no_trigger_{nb_triggers}.csv")


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
    A_init = np.copy(M[0])

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

    return M, A_init, cycles_list


# **************** #
# Input Generation #
# **************** #

def generate_input(input_dim=1, input_size=300):

    # U = random_input(dim=input_dim, length=input_size) # XXX
    U = poisson_input(lamda=5, dim=input_dim, length=input_size)

    return U

# ********** #
# Simulation #
# ********** #

def run_simulation(U, M, A_init, cycles_list):

    eta = eta_in

    for nb_iter in tqdm(range(len(U) - 2)):

        # Extract next input segment
        current_input = {0: U[nb_iter]} # , 1: U[nb_iter + 1]} # XXX

        # Simulate network behavior and apply STDP rule
        x = M[-1] # current state
        _, _, M = simulation(M[0], M[1], M[2], M[3], x, 
                                    current_input,
                                    epoch=len(current_input), 
                                    stdp="off"
                                    ) # XXX

        # Compute cycles for the current network state
        A_new = netM_to_aut(M)
        dico_cycles = get_simple_cycles(A_new)
        # get SCC including current state
        for SCC in dico_cycles.keys():
            if M[4] in SCC:
            # SCC = largest_list(list(dico_cycles.keys())) # XXX old version
             _, n = dico_cycles[SCC]

        # Update cycle tracking and STDP parameters # XXX no more STDP
        # stack_operation(cycles_list, memory_length, n)
        # min_cycles, max_cycles = min(cycles_list), max(cycles_list)
        # eta = linear(min_cycles, eta_in, max_cycles, eta_in / 20.0, n)


        # Logging
        if verbose:
            print(current_input, M[4], M[0], M[1], cycles_list, eta)

        # Write results to file
        with open(output_file, "a") as f:
            f.write(f"{memory_length}, {n}, {min_cycles}, {max_cycles}, {eta}, {tick}\n")




import numpy as np
import math
import random



np.random.seed(95)
verbose = False
eta_in = 0.15
plumb = 1
bounds = (-0.1999, 0.5999)
noise = (-0.1, 0.5)
input_size = 3000 # Adjust as necessary
memory, memory_length = 149, 1
input_dim = 1
nb_triggers = 19
trigger_length = 10

cwd = os.getcwd()
output_file = os.path.join(cwd, "runs", f"GSP_no_trigger_{nb_triggers}.csv")


# ***************************** #
# Initialization of Simulations #
# ***************************** #

def attractor_energy(net_matrices):
    """XXX TO DO"""
    cycle_d = get_current_attractor(net_matrices)  # attractor containing current state x
    _, cycles = list(cycle_d.items())[0] # get the only one key:value
    n = cycles[1]

    return -n # minimize energy

def generate_candidate(current_solution, noise=(-0.1, 0.1)):
    # XXX update the function to add bounds, i.e clipping values, etc.
    """
    Generates a candidate solution that is "compatible" with the current state.
    The candidate solution is a network in matricial form M, where the synapses M[0] have been distorted.
    This modified network yields a new attractor dynamics. If the current state x is not in an 

    """
    A = current_solution[0]
    valid = False

    while not valid:

        A_new = distort(A, noise=noise)
        current_solution[0] = A_new
        cycle_d = get_current_attractor(current_solution)

        if cycle_d != None:
            valid = True

    return current_solution

def simulated_annealing(initial_solution, func, max_iterations, initial_temp, cooling_rate):

    current_solution = initial_solution
    current_energy = func(current_solution)

    best_solution = current_solution[:]
    best_energy = current_energy
    
    temperature = initial_temp

    for iteration in range(max_iterations):
        # Generate a new candidate solution
        # Ensure candidate is within bounds: clip values???
        candidate = generate_candidate(current_solution=current_solution)
        
        candidate_energy = func(candidate)
        
        # Calculate the energy difference
        energy_difference = candidate_energy - current_energy
        
        # Accept candidate if it's better, or with a probability that decreases with temperature
        if energy_difference < 0 or math.exp(-energy_difference / temperature) > random.random():
            current_solution = candidate[:]
            current_energy = candidate_energy
            
            # Check if this is the best solution found so far
            if current_energy < best_energy:
                best_solution = current_solution[:]
                best_energy = current_energy
        
        # Cool down the temperature
        temperature *= cooling_rate
        
        # Optionally print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Best Energy: {best_energy}, Temperature: {temperature}")

    return best_solution, best_energy, temperature


nb_iter = 100         # Number of iterations
temperature = 10.0   # Initial temperature
cooling_rate = 0.99  # Cooling rate (close to 1 means slower cooling)

if __name__ == "__main__":

    # Generate input
    U = poisson_input(lamda=3, dim=input_dim, length=input_size)

    # Generate BGT network and matrix
    N = (input_nodes_N, nodes_N, edges_N)
    M = net_to_matrix(N)

    for nb_iter in tqdm(range(len(U) - 2)):

        # Extract next input segment
        current_input = {0: U[nb_iter]}

        # Simulate network
        x = M[-1] # current state
        _, _, M = simulation(M[0], M[1], M[2], M[3], x, 
                             current_input,
                             epoch=len(current_input), 
                             stdp="off"
                             ) # XXX

        # Run simulated annealing
        M, best_energy, temperature = simulated_annealing(M, attractor_energy,
                                                          nb_iter, temperature,
                                                          cooling_rate)

    
#!/opt/local/bin/python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

source_path = str(Path(__file__).parent.parent)
sys.path.append(source_path)

import numpy as np
from source.attractors import *
from source.network import *


def attractor_energy(net_matrices):
    """In the 'static' case (i.e. without STDP), 
    gets largest attractor of the network"""
    max_cycle = get_largest_attractor(net_matrices)
    try:
        n = list(max_cycle.values())[0][-1]
    except:
        n = 0  # candidate solution might have no attractors

    return -n  # minimize energy

def generate_candidate(current_solution, noise=0.1, state_aware=False):
    """
    Generates a candidate solution that is "compatible" with the current state.
    The candidate solution is a network in matricial form M, 
    where the synapses M[0] have been distorted.
    In the case of state_aware=True, the current state x must belong 
    to some non-degenerate attractor of the candidate solution.
    The candidate solution yields a new attractor dynamics. 
    """
    A = current_solution[0]

    if state_aware:

        valid = False

        while not valid:

            A_new = distort(A, noise=noise)
            current_solution[0] = A_new
            cycle_d = get_current_largest_attractor(current_solution)

            if cycle_d != None:
                valid = True
    else:
        
        A_new = distort(A, noise=noise)
        current_solution[0] = A_new

    return current_solution


def simulated_annealing(initial_solution, func, 
                        max_iterations, initial_temp, cooling_rate,
                        noise=0.1, state_aware=False, verbose=True):

    current_solution = initial_solution
    current_energy = func(current_solution)
    best_solution = current_solution[:]
    best_energy = current_energy

    best_energies = []
    best_energies.append(best_energy)
        
    temp = initial_temp
    temps = []
    temps.append(temp)

    for iteration in range(max_iterations):
        # Generate a new candidate solution
        # Ensure candidate is within bounds: clip values???
        candidate = generate_candidate(current_solution=current_solution, 
                                       noise=noise, state_aware=state_aware)
        candidate_energy = func(candidate)
        
        # Calculate the energy difference
        energy_diff = candidate_energy - current_energy
        
        # Accept candidate if it's better or
        # with a probability that decreases with temperature
        if energy_diff < 0 or np.exp(-energy_diff / temp) > np.random.rand():
            current_solution = candidate[:]
            current_energy = candidate_energy

            # Check if this is the best solution found so far
            if current_energy < best_energy:
                best_solution = current_solution[:]
                best_energy = current_energy

        best_energies.append(best_energy)
        
        # Cool down the temperature
        temp *= cooling_rate
        temps.append(temp)
        
        # Optionally print progress every 100 iterations
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}    \
                    Best Energy: {best_energy} \
                    Temperature: {temp}")

    return best_energies, temps, best_solution

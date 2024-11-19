#!/opt/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

source_path = str(Path(__file__).parent.parent)
# source_path = os.path.join(source_path, "source")
sys.path.append(source_path)

from tqdm import tqdm

from source.attractors import *
from source.network import *


def attractor_energy(net_matrices):
    """In the 'static' case (i.e. without STDP), gets largest attractor of the network"""
    max_cycle = get_largest_attractor(net_matrices)
    try:
        n = list(max_cycle.values())[0][1]
    except:
        n = 0  # candidate solution might have no attractors

    return -n  # minimize energy

def generate_candidate(current_solution, noise=(-0.1, 0.1), state_aware=False):
    # XXX update the function to add bounds, i.e clipping values, etc.
    """
    Generates a candidate solution that is "compatible" with the current state.
    The candidate solution is a network in matricial form M, where the synapses M[0] have been distorted.
    This modified network yields a new attractor dynamics. If the current state x is not in an XXX
    """
    A = current_solution[0]

    if state_aware:

        valid = False

        while not valid:

            A_new = distort(A, noise=noise)
            current_solution[0] = A_new
            cycle_d = get_current_attractor(current_solution)

            if cycle_d != None:
                valid = True
    else:
        
        A_new = distort(A, noise=noise)
        current_solution[0] = A_new

    return current_solution


def simulated_annealing(initial_solution, func, max_iterations, initial_temp, cooling_rate):

    current_solution = initial_solution
    current_energy = func(current_solution)

    best_solution = current_solution[:]
    best_energy = current_energy
    
    temperature = initial_temp

    for iteration in tqdm(range(max_iterations)):
        # Generate a new candidate solution
        # Ensure candidate is within bounds: clip values???
        candidate = generate_candidate(current_solution=current_solution)
        
        candidate_energy = func(candidate)
        
        # Calculate the energy difference
        energy_difference = candidate_energy - current_energy
        
        # Accept candidate if it's better, or with a probability that decreases with temperature
        if energy_difference < 0 or np.exp(-energy_difference / temperature) > random.random():
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


if __name__ == "__main__":
    pass
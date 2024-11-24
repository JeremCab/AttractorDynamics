#!/usr/bin/env python

# ********* #
# Libraries #
# ********* #

import os
import numpy as np
import pickle
from tqdm import tqdm

from source.bgt_network import *
from source.network import *
from source.attractors import *
from optim.simulated_annealing import *

# ********** #
# Parameters #
# ********** #

# Parameters
# Modes:
# gp        -> global plasticity
# stdp      -> spike-timing ddependent plalsticity
# stdp-gp   -> combined mode
mode = "stdp-gp"  # "gp" "stdp" "stdp-gp"

seed = 42
np.random.seed(seed)

input_length = 1000 # 3000
trigger_length = 20
nb_triggers = 20
temperature = 10.0    # Initial temperature
cooling_rate = 0.995  # Cooling rate (close to 1 means slower cooling)

eta = 0.15
plumb = 1
bounds = (-0.4999, 1.4999)  # clipping values for STDP
noise = 0.3                 # noise for SA 

cwd = os.getcwd()
results_folder = "runs"
results_folder = os.path.join(cwd, results_folder)
results_file = f"sim_{mode}_{input_length}_{trigger_length}_{nb_triggers}.pkl"
results_file = os.path.join(results_folder, results_file)

# ************** #
# Initialization #
# ************** #

# BGT Network
network = (input_nodes_N, nodes_N, edges_N)
M = net_to_matrix(network)
# Apply random distortion to weight matrix M[0]
# distort(M[0], noise=noise)

# Input
U, ticks = generate_input(input_dim=M[1].shape[0], 
                          input_length=input_length, 
                          mode="poisson", 
                          lamda=2, 
                          triggers=True, 
                          nb_triggers=nb_triggers,
                          trigger_length=trigger_length)

# ********** #
# Simulation #
# ********** #

if __name__ == "__main__":

    print("*" * 100)
    print("Simulation started...")

    # *** GP only ***
    if mode == "gp":

        sim = simulated_annealing(initial_solution=M, 
                                func=attractor_energy, 
                                max_iterations=input_length, 
                                initial_temp=temperature, 
                                cooling_rate=cooling_rate, 
                                noise=noise)

        print("Simulation done.")

        with open(results_file, "wb") as fh:
            pickle.dump(sim, fh)
        
        print("Results saved.")

    # *** STDP only ***
    elif mode == "stdp":

        _, synapses, _ = simulation(M[0], M[1], M[2], M[3], M[4], 
                                    U,
                                    epoch=input_length, 
                                    stdp=[M[0], eta, plumb, bounds])
        
        print("Simulation done.")
            
        nb_attractors = get_nb_attractors(synapses, M)

        with open(results_file, "wb") as fh:
            pickle.dump(nb_attractors, fh)
        
        print("Results saved.")
        
    # *** combined STDP and GP ***
    if mode == "stdp-gp":
        
        t0 = 0
        input_blocks = []

        for t in ticks:
            t0_t1 = list(range(t0, t))
            input_blocks.append(t0_t1)
            t0 = t

        nb_attractors = []

        # 1st input block
        # STDP (initial)
        b = input_blocks[0]
        U_b = dict([(k, v) for k, v in U.items() if k in b])
        print("STDP in process (initial)...")
        A = M[0].copy() # dummy var for testing XXX
        _, synapses, M = simulation(M[0], M[1], M[2], M[3], M[4], 
                                    U_b,
                                    epoch=len(b),     # len(b) iterations
                                    stdp=[M[0], eta, plumb, bounds])

        attrs = get_nb_attractors(synapses, M)
        nb_attractors.extend(attrs)
        print("    -> STDP   changed M[0]:", (A != M[0]).any())

        # subsequent input blocks
        for i, b in enumerate(input_blocks[1:]):

            b0 = b[:trigger_length]  # input prefix: trigger_length time steps
            b1 = b[trigger_length:]  # input suffix: remaining time steps
            U_b0 = dict([(k, v) for k, v in U.items() if k in b0])
            U_b1 = dict([(k, v) for k, v in U.items() if k in b1])

            # GP (input block prefix)
            print("GP in process...")
            A = M[0].copy() # dummy var for testing XXX
            sim = simulated_annealing(initial_solution=M, 
                                    func=attractor_energy, 
                                    max_iterations=len(b0),  # 1 iteration ??? XXX
                                    initial_temp=temperature, 
                                    cooling_rate=cooling_rate, 
                                    noise=noise, 
                                    state_aware=True, 
                                    verbose=False)
                
            best_energies, temps, M = sim
            best_energy = max(best_energies)
            temperature = temps[-1]
            nb_attractors.extend(best_energies)
            print("    -> GP   best_energies", best_energies)
            print("    -> GP   changed M[0]:", (A != M[0]).any())

            # STDP (input block sufix)
            print("STDP in process...")
            A = M[0].copy() # dummy var for testing XXX
            _, synapses, M = simulation(M[0], M[1], M[2], M[3], M[4], 
                                        U_b1,
                                        epoch=len(b1),     # len(b1) iterations xxx
                                        stdp=[M[0], eta, plumb, bounds])
            
            attrs = get_nb_attractors(synapses, M)
            nb_attractors.extend(attrs)
            print("    -> STDP   best_energies", best_energies)
            print("    -> STDP   changed M[0]:", (A != M[0]).any())
            print("    -> synaptic steps", synapses.shape[2])
            print("    -> CURRENT ATTRACTORS\n", nb_attractors)
            # print(f"Step {i}: (M[0]", M[0])
            # attractors_lengths = get_nb_attractors(synapses, M)

        # print("Simulation done.")

        with open(results_file, "wb") as fh:
            pickle.dump([nb_attractors, ticks], fh)
    
        print("Results saved.")

#!/usr/bin/env python

# ********* #
# Libraries #
# ********* #

import os
import numpy as np
import pickle
# from tqdm import tqdm

import argparse

from source.bgt_network import *
from source.network import *
from source.attractors import *
from optim.simulated_annealing import *

# ********** #
# Parameters #
# ********** #

# Create argument parser
parser = argparse.ArgumentParser(description="Run experiment with varying parameters.")

# Add arguments for each parameter
parser.add_argument("--mode", type=str, default="stdp", choices=["gp", "stdp", "stdp-gp"],
                    help="Choose the mode: 'gp', 'stdp', or 'stdp-gp'. Default is 'stdp'.")
parser.add_argument("--input_length", type=int, default=1001, help="Length of input sequence. Default is 1001.")
parser.add_argument("--trigger_length", type=int, default=50, help="Length of trigger. Default is 50.")
parser.add_argument("--nb_triggers", type=int, default=10, help="Number of triggers. Default is 10.")
parser.add_argument("--seed", type=int, default=42, help="Random seed. Default is 42.")
parser.add_argument("--temperature", type=float, default=10.0, help="Initial temperature. Default is 10.0.")
parser.add_argument("--cooling_rate", type=float, default=0.995, help="Cooling rate. Default is 0.995.")
parser.add_argument("--eta", type=float, default=0.025, help="STDP learning rate eta. Default is 0.025.")
parser.add_argument("--plumb", type=float, default=1.0, help="STDP plumb value. Default is 1.")
parser.add_argument("--bounds", type=str, default="-0.4999,1.4999", 
                    help="STDP bounds as a string. Default is 'minus0.4999,1.4999' ('-' sign causes problems).")
parser.add_argument("--noise", type=float, default=0.3, help="Noise for SA. Default is 0.3.")

# Parse the arguments
args = parser.parse_args()

# Set parameters
seed = args.seed
mode = args.mode
input_length = args.input_length
trigger_length = args.trigger_length
nb_triggers = args.nb_triggers
temperature = args.temperature
cooling_rate = args.cooling_rate
eta = args.eta
plumb = args.plumb
bounds = [float(x.replace("minus", "-")) for x in args.bounds.split(",")]  # string -> tuple of floats
noise = args.noise

# Set random seed
np.random.seed(seed)

# Results file
cwd = os.getcwd()
results_folder = "runs"
results_folder = os.path.join(cwd, results_folder)

# results_file = f"sim_{mode}_{input_length}_{trigger_length}_{nb_triggers}_seed{seed}.pkl"  # XXX
results_file = f"sim_{mode}_{input_length}_{trigger_length}_{nb_triggers}_seed{seed}_eta{eta}.pkl"

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

print("ticks:", ticks)

# ********** #
# Simulation #
# ********** #

if __name__ == "__main__":

    print("*" * 100)
    print("Simulation started...")

    # *** GP only *** #
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
        
        print(f"Results saved: {results_file}")

    # *** STDP only *** #
    elif mode == "stdp":

        _, synapses, _ = simulation(M[0], M[1], M[2], M[3], M[4], 
                                    U,
                                    epoch=input_length, 
                                    stdp=[M[0], eta, plumb, bounds])
        
        print("Simulation done.")

        nb_attractors = get_nb_attractors(synapses, M)

        synapses_l = [synapses[:, :, i] for i in range(synapses.shape[2])]

        results = (nb_attractors, synapses_l)

        with open(results_file, "wb") as fh:
            pickle.dump(results, fh)
        
        print(f"Results saved: {results_file}")
        
    # *** combined STDP and GP *** #
    if mode == "stdp-gp":
        
        t0 = 0
        input_blocks = []

        for t in ticks:
            t0_t1 = list(range(t0, t))
            input_blocks.append(t0_t1)
            t0 = t
        last_block = list(range(t0, input_length))
        input_blocks.append(last_block)

        synapses_l = []
        nb_attractors = []

        # 1st input block
        # STDP (initial)
        b = input_blocks[0]
        U_b = dict([(k, v) for k, v in U.items() if k in b])
        print("STDP in process (initial)...")
        A = M[0].copy() # dummy var for checking syn. change
        history, synapses, M = simulation(M[0], M[1], M[2], M[3], M[4], 
                                    U_b,
                                    epoch=len(b),     # len(b) iterations
                                    stdp=[M[0], eta, plumb, bounds])

        attrs = get_nb_attractors(synapses, M)
        nb_attractors.extend(attrs)

        synapses = [synapses[:, :, i] for i in range(synapses.shape[2])]
        synapses_l.extend(synapses)

        print("    -> STDP changed M[0]:", (A != M[0]).any())

        # subsequent input blocks
        for i, b in enumerate(input_blocks[1:]):

            b0 = b[:trigger_length]  # input prefix: trigger_length time steps
            b1 = b[trigger_length:]  # input suffix: remaining time steps
            U_b0 = dict([(k, v) for k, v in U.items() if k in b0])
            U_b1 = dict([(k, v) for k, v in U.items() if k in b1])

            # GP (input block prefix)
            print(f"GP in process..(temparture: {temperature})")
            A = M[0].copy() # dummy var for checking syn. change
            sim = simulated_annealing(initial_solution=M, 
                                      func=attractor_energy, 
                                      max_iterations=len(b0),  # b0 steps
                                      initial_temp=temperature, 
                                      cooling_rate=cooling_rate, 
                                      noise=noise, 
                                      state_aware=True, 
                                      verbose=False)

            best_energies, temps, M = sim

            # best_energy = max(best_energies)
            # temperature = temps[-1]  # if commented, temperature reset at every new input block XXX
            best_energies = list(-np.array(best_energies))
            nb_attractors.extend(best_energies)

            synapses = [A]*(len(b0) - 1) + [M[0]]
            synapses_l.extend(synapses)

            print("    -> GP   changed M[0]:", (A != M[0]).any())
            # print("    -> CURRENT ATTRACTORS\n", nb_attractors)

            # STDP (input block sufix)
            print("STDP in process...")
            A = M[0].copy() # dummy var for checking syn. change

            # NOTE: for a proper functioning of the `simulation` function, 
            # the ticks of U_b1 need to be rescaled between 0 and len(U_b1).
            U_b1 = {kk: U_b1[k] for kk, k in enumerate(sorted(U_b1.keys()))}  # rekey U_b1

            history, synapses, M = simulation(M[0], M[1], M[2], M[3], M[4], 
                                              U_b1,
                                              epoch=len(b1),  # b1 steps
                                              stdp=[M[0], eta, plumb, bounds])

            attrs = get_nb_attractors(synapses, M)[1:] # first already computed in previous step
            nb_attractors.extend(attrs)

            synapses = [synapses[:, :, i] for i in range(synapses.shape[2])]
            synapses_l.extend(synapses)

            print("    -> STDP changed M[0]:", (A != M[0]).any())
            # print("    -> CURRENT ATTRACTORS\n", nb_attractors)

        # print("Simulation done.")

        with open(results_file, "wb") as fh:
            pickle.dump((nb_attractors, ticks, synapses_l), fh)

        print(f"Simulation ended.")
        print(f"Results saved: {results_file}")

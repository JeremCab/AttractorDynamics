# -*- coding: utf-8 -*-

import numpy as np

# ******************* #
# ORIGINAL PARAMETERS #
# ******************* #

# List of weights and parameters for the network
weights = [
    None,  # Placeholder so indices match with network edge positions
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    2.0, 2.0, 2.0, -0.5, -0.5, -0.5, -0.5, -0.5, -1.0, -0.5, -0.5,
    0.5, 1.0, 0.5, 0.5, 0.5, 0.5,
    0.0, 0.0,  # Interactive weights (31 and 32)
    1.0        # Threshold (33)
]

# ************* #
# THE NETWORK   #
# ************* #

# Node definitions
nodes_BGT = ['SupCo', 'Thala', 'ThRtN', 'SNRet', 'SThNu', 'GlbPe', 'D2MSN', 'D1MSN', 'CerCx']
nodes_BGT_bis = ['input'] + nodes_BGT
nodes_N = list(range(1, 1+len(nodes_BGT)))  # Node indices for internal cells
input_nodes_N = [0]                         # only 1 input

# Edge connections with weights (interactive connections, then network connections)
edges_N = [
    [(1, 0), weights[31]], [(9, 0), weights[32]],  # Interactive connections
    [(0, 1), weights[1]], [(0, 2), weights[2]],
    [(1, 2), weights[3]], 
    [(2, 3), weights[4]], [(2, 5), weights[5]], [(2, 6), weights[6]], [(2, 7), weights[7]], [(2, 8), weights[8]], [(2, 9), weights[9]],
    [(3, 2), weights[10]], 
    [(4, 1), weights[11]], [(4, 2), weights[12]], [(4, 3), weights[13]],
    [(5, 4), weights[14]], [(5, 6), weights[15]], [(5, 9), weights[16]],
    [(6, 3), weights[17]], [(6, 4), weights[18]], [(6, 5), weights[19]], [(6, 7), weights[20]], [(6, 8), weights[21]],
    [(7, 6), weights[22]], 
    [(8, 4), weights[23]], [(8, 6), weights[24]],
    [(9, 1), weights[25]], [(9, 2), weights[26]], [(9, 3), weights[27]], [(9, 5), weights[28]], [(9, 7), weights[29]], [(9, 8), weights[30]]
    # [(9, 10), "1"],                   # remove node 10
    # [(10,2), "0.5"], [(10,4), "0.5"]  # remove node 10
]



# *********************** #
# Example Usage and Tests #
# *********************** #

# Uncomment below for testing

# from RNN_simulator import simulation

# # Matrix A: weight matrix for internal connections
# dim1 = len(nodes_N)
# A = np.zeros((dim1, dim1))
# for (src, dest), weight in edges_N:
#     if src > 0 and dest > 0:  # Internal connections
#         A[src - 1, dest - 1] = weight

# # Matrix B1: weight matrix for input-to-internal connections
# dim2 = len(input_nodes_N)
# B1 = np.zeros((dim2, dim1))
# for (src, dest), weight in edges_N:
#     if src < dim2:  # Input cells are the first in `input_nodes_N`
#         B1[src - 1, dest - 1] = weight

# # Matrix B2: weight matrix for internal-to-input connections
# B2 = np.zeros((dim1, dim2))
# for (src, dest), weight in edges_N:
#     if dest < dim2:  # Input cells are the first in `input_nodes_N`
#         B2[src - 1, dest - 1] = weight

# # Vector b: bias vector (initialized to zero)
# b = np.zeros((A.shape[0], 1))

# # Vector x: Initial state (null initial state)
# x = np.zeros((A.shape[0], 1))

# # Input dictionary U: Random binary inputs of size 100
# input_sequence = np.random.randint(0, 2, 101)
# U = {i: np.array([[bit]]) for i, bit in enumerate(input_sequence)}

# # Simulation
# inputs_states, synapses, matrices = simulation(A, B1, B2, b, x, U, epoch=101)

# print("last inputs and states\n", inputs_states[:,-20:])
# print("synapses\n", matrices[0])
# print((matrices[0] == synapses[:, :, 0]).all())
# print((matrices[0] == synapses[:, :, -1]).all())
# print((matrices[0] == synapses[:, :, 55]).all())
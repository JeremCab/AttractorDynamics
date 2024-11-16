# -*- coding: utf-8 -*-

import numpy as np

if __name__ != "__main__":
    from source.rnn_simulator import simulation
    import source.TarjanJohnson as tj
else:
    from rnn_simulator import simulation
    import TarjanJohnson as tj

# **************************** #
# Encoding and Decoding Binary #
# **************************** #

def code(binary_list):
    """
    Encodes a binary list into an integer.
    Args:
        binary_list (list of int): Binary list to encode.
    Returns:
        int: Encoded integer.
    """
    binary_list = binary_list[::-1]  # Reverse list
    return sum(x * (2 ** i) for i, x in enumerate(binary_list))


def decode(n, k):
    """
    Decodes an integer into a binary list of length k.
    Args:
        n (int): Integer to decode.
        k (int): Length of the output binary list.
    Returns:
        list of int: Binary representation.
    """
    bin_str = bin(n)[2:].zfill(k)
    return [int(bit) for bit in bin_str]


# ******************************** #
# Matrix Representation of Network #
# ******************************** #

def net_to_matrix(Network):
    """
    Constructs matrix representations for a given network.
    Args:
        Network (tuple): Network defined as (input_cells, cells, connections).
    Returns:
        list: List of matrices [A, B1, B2, b, x] representing the network.
    """
    input_cells, cells, connections = Network
    n_inputs, n_cells = len(input_cells), len(cells)
    
    # Initialize matrices
    A = np.zeros((n_cells, n_cells))
    B1 = np.zeros((n_inputs, n_cells))
    B2 = np.zeros((n_cells, n_inputs))
    b = np.zeros((n_cells, 1))
    x = np.zeros((n_cells, 1))

    # Fill matrices based on connections
    for (src, dest), weight in connections:
        if src == "env":        # Bias connection from environment
            b[dest - n_inputs] = weight
        elif src < n_inputs:    # Input connections
            B1[src][dest - n_inputs] = weight
        elif dest < n_inputs:   # Interactive connections
            B2[src - n_inputs][dest] = weight
        else:                   # Regular connections
            A[src - n_inputs][dest - n_inputs] = weight

    return [A, B1, B2, b, x]


def matrix_to_net(A, B1, B2, b):
    """
    Retrieves a network from its matrix representation.
    Args:
        A (np.ndarray): Internal connections matrix.
        B1 (np.ndarray): Input to internal connections matrix.
        B2 (np.ndarray): Internal to input connections matrix.
        b (np.ndarray): Bias connections matrix.
    Returns:
        tuple: Network as (input_cells, cells, connections).
    """
    n_inputs = B1.shape[0]
    n_cells = A.shape[0]
    
    input_cells = list(range(n_inputs))
    cells = list(range(n_inputs, n_cells + n_inputs))
    connections = []

    # Collect connections from matrices
    for i in range(B1.shape[0]):
        for j in range(B1.shape[1]):
            connections.append([(i, j + n_inputs), B1[i][j]])
    for i in range(B2.shape[0]):
        for j in range(B2.shape[1]):
            connections.append([(i + n_inputs, j), B2[i][j]])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            connections.append([(i + n_inputs, j + n_inputs), A[i][j]])
    for i in range(b.shape[0]):
        connections.append([("env", i + n_inputs), b[i][0]])

    return (input_cells, cells, connections)

# ************************** #
# Dictionary form for graphs #
# ************************** #

def dico_form(G):
    """
    Transforms a graph into its dictionary representation: key: value = node: successor_list
    Args:
        G (tuple): Graph as (nodes, edges).
    Returns:
        dict: Dictionary form of the graph.
    """
    graph_dict = {}
    for (src, dest), _ in G[1]:
        graph_dict.setdefault(src, []).append(dest)
    return graph_dict


# **************************** #
# Boolean Network to Automaton #
# **************************** #

def net_to_aut(N):
    """
    Converts a Boolean network into an automaton.
    Args:
        N (tuple): Network in the form (input_cells, cells, connections).
    Returns:
        tuple: Automaton as (nodes, edges).
    """
    A, B1, B2, b, _ = net_to_matrix(N)
    n_cells, n_inputs = A.shape[0], B1.shape[0]
    
    aut_nodes = list(range(2 ** n_cells))
    aut_edges = []

    for x_A in aut_nodes:
        x = np.array([[bit] for bit in decode(x_A, n_cells)])                   # network state (vector)
        
        for y in range(2 ** n_inputs):
            u = np.array([[bit] for bit in decode(y, n_inputs)])                # input (vector)
            U = {0: u}
            x_plus = simulation(A, B1, B2, b, x, U, epoch=2)[0][n_inputs:, -1]  # next network state (vector)
            x_plus_A = code([int(i) for i in x_plus])
            aut_edges.append([(x_A, x_plus_A), y])

    return (aut_nodes, aut_edges)

def netM_to_aut(M):
    """
    Converts a Boolean network given given in matricial form M into an automaton.
    Args:
        M (tuple): Network in the matricial form M = (A, B1, B2, b, x).
    Returns:
        tuple: Automaton as (nodes, edges).
    """
    A, B1, B2, b, _ = M
    n_cells, n_inputs = A.shape[0], B1.shape[0]
    
    aut_nodes = list(range(2 ** n_cells))
    aut_edges = []

    for x_A in aut_nodes:
        x = np.array([[bit] for bit in decode(x_A, n_cells)])                   # network state (vector)
        
        for y in range(2 ** n_inputs):
            u = np.array([[bit] for bit in decode(y, n_inputs)])                # input (vector)
            U = {0: u}
            x_plus = simulation(A, B1, B2, b, x, U, epoch=2)[0][n_inputs:, -1]  # next network state (vector)
            x_plus_A = code([int(i) for i in x_plus])
            aut_edges.append([(x_A, x_plus_A), y])

    return (aut_nodes, aut_edges)

# ********************** #
# Cycles in an Automaton #
# ********************** #

def get_simple_cycles(A):
    """
    Computes simple cycles in an automaton.
    Args:
        A (tuple): Automaton as (nodes, edges).
    Returns:
        dict: Dictionary with strongly connected components (SCCs) and their cycles.
    """
    result = {}
    A_dico = dico_form(A)

    for scc in tj.SCC(A_dico):
        if tj.is_an_SCC(scc) or tj.check_scc_1(scc, A_dico):
            scc_dico = {node: [n for n in A_dico[node] if n in scc] for node in scc}
            cycles = tj.get_elementary_cycles(scc_dico)
            result[scc] = (cycles, len(cycles))

    return result


def largest_list(list_of_lists):
    """
    Finds the largest list within a list of lists.
    Args:
        list_of_lists (list of list): List containing multiple lists.
    Returns:
        list: Largest list by length.
    """
    return max(list_of_lists, key=len, default=[])

# ************************ #
# Attractors in an Network #
# ************************ #

def get_attractors(matrices):
    """
    Computes attractors of a network given in matricial form.
    Attractors are returned as a dico of cycles.
    """
    A, B1, B2, b, _ = matrices
    N = matrix_to_net(A, B1, B2, b)
    A = net_to_aut(N)
    dico_cycles = get_simple_cycles(A)
    return dico_cycles

def get_current_attractor(matrices):
    """
    Given a network N given in matricial form M, where the current state x is given by M[4], 
    get current attractor N containing state x, if it exists.
    The current attractor is returned as a dico of the form  {SCC : (tuple_of_cycles, number_of_cycles)}
    """
    x = matrices[4]
    x_code = int(code(x.reshape(-1)))

    dico_cycles = get_attractors(matrices)

    for SCC in dico_cycles.keys():
        if x_code in SCC:
            return {SCC: dico_cycles[SCC]}
        return None



# *********************** #
# Example Usage and Tests #
# *********************** #

if __name__ == "__main__":

    # Graph dictionary
    V = [1,2,3,4]
    E = [ [(1,3), 0.3], [(1,5), 0.1],  [(2,1), 0.7], [(2,2), 0.9], [(2,4), 0.5], 
        [(3,4), 0.2], [(3,5), 0.7], [(4,1), 0.8], [(4,2), 0.1], [(4,3), 0.8] ]
    G = (V, E)
    print(G)
    D = dico_form(G)
    print(D)

    # From network to matrix and back
    input_cells = [0, 1]
    cells = [2,3,4,5]
    connections = [	[(0,3), 0.7], # input
                    [(0,4), 0.2], # input
                    [(1,2), 0.4], # input
                    [(1,5), 0.9], # input
                    [(2,3), 0.2],
                    [(3,2), 0.3],
                    [(3,3), 0.3],
                    [(3,4), 0.3],
                    [(4,0), 0.7], # interactive
                    [(4,5), 0.3],
                    [(5,0), 0.4], # interactive
                    [(5,1), 0.3], # interactive
                    [(5,2), 0.8],
                    [(5,3), 0.1],
                    [("env",3), 0.6], # bias
                    [("env",5), 0.7], # bias
                    ]
    N1 = (input_cells, cells, connections)
    # print("N1", N1)
    M = net_to_matrix(N1)
    # print("A", M[0])
    # print("B1", M[1])
    # print("B2", M[2])
    # print("C", M[3])
    print("x", M[4])

    N2 = matrix_to_net(M[0], M[1], M[2], M[3])
    # print("N2", N2)
    # print("input cells")
    # print(N1[0] == N2[0])
    # print("cells")
    # print(N1[1] == N2[1])
    # print("connections")

    b = True
    for e in N1[2]:
        if e not in N2[2]:
            print(e)
            b = False
    print(b)
    b = True
    for e in N2[2]:
        if e not in N1[2] and e[1] != 0:
            print(e)
            b = False
    print(b)
    # OK: N1 et N2 are identical

    # Conputing attractors
    input_cells = [0, 1]
    cells = [2,3,4,5]
    connections = [[(0,3), 0.7],  # input
                    [(0,4), 0.2], # input
                    [(1,2), 0.4], # input
                    [(1,5), 0.9], # input
                    [(2,3), 0.2],
                    [(3,2), 0.3],
                    [(3,3), 0.3],
                    [(3,4), 0.3],
                    [(4,0), 0.7], # interactive
                    [(4,5), 0.3],
                    [(5,0), 0.4], # interactive
                    [(5,1), 0.3], # interactive
                    [(5,2), 0.8],
                    [(5,3), 0.1],
                    [("env",3), 0.6], # bias
                    [("env",5), 0.7]  # bias
                    ]

    print("\nNetwork N")
    N = (input_cells, cells, connections)
    print(N)
    print("\nComputation of automaton A")
    A = net_to_aut(N)
    print(A)

    print("\nCycles of A")
    cycles = get_simple_cycles(A)
    print(cycles)
    print("\Attractors of N")
    attractors = get_attractors(M)
    print(attractors)
    print("\Attractor of N containing state x")
    M[4] = np.array([1., 1., 0., 1.]).reshape(-1, 1)
    print("x", M[4], "code of x", int(code(M[4].reshape(-1,)))) # change state x
    print(get_current_attractor(M))

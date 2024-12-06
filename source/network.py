#!/opt/local/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

# *********************** #
# 3 Preliminary Functions #
# *********************** #

def stack_operation(lst: list, max_length: int, x) -> None:
    """Manage list length by appending new element and keeping its length within max_length."""
    if len(lst) < max_length:
        lst.append(x)
    else:
        del lst[:len(lst) - max_length + 1]
        lst.append(x)

def linear(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    """Compute y on a line through (x1, y1) and (x2, y2) at a specific x value."""
    try:
        slope = (y2 - y1) / (x2 - x1)
        intercept = (y1 * x2 - y2 * x1) / (x2 - x1)
        return slope * x + intercept
    except ZeroDivisionError:
        return y1

# ********************************* #
# Perturbation of the Weight Matrix #
# ********************************* #

def distort(matrix: np.ndarray, 
            noise: float = 0.1, 
            min_max: tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
    """
    Randomly distort non-zero weights in a matrix within a noise range.
    The signs of the weights are preserved: a positive (negative) weight stays so.
    Clip values between min and max.
    """
    # Create a mask for non-zero elements
    non_zero_mask = matrix != 0

    # Generate random noise for elements in the matrix within specified range
    noise_matrix = np.random.uniform(-noise, noise, size=matrix.shape)
    noise_matrix = noise_matrix * non_zero_mask

    # Apply noise only to non-zero elements
    distorted_matrix = matrix + noise_matrix

    # Keep same signs
    diff_signs = (distorted_matrix * matrix) < 0
    distorted_matrix[diff_signs] = matrix[diff_signs]

    # Clip values
    mini, maxi = min_max[0], min_max[1]
    distorted_matrix = np.maximum(distorted_matrix, non_zero_mask * mini)
    distorted_matrix = np.minimum(distorted_matrix, non_zero_mask * maxi)

    return distorted_matrix

# ************************ #
# Generating Random Inputs #
# ************************ #

def random_input(dim: int = 1, length: int = 10) -> dict[int, np.ndarray]:
    """Generate a random Boolean input stream of given dimension and length."""
    return {t: np.random.randint(2, size=(dim, 1)) for t in range(length)}

# ************************** #
# Generating Periodic Inputs #
# ************************** #

def periodic_input(dim: int = 1, length: int = 10, times: int = 10) -> dict[int, np.ndarray]:
    """Generate a random Boolean input stream repeated a specified number of times."""
    inputs = {t: np.random.randint(2, size=(dim, 1)) if t < length else np.zeros((dim, 1)) for t in range(length * 2)}
    for t in range(length * 2, length * 2 * times):
        inputs[t] = inputs[t - length * 2]
    return inputs

# ************************* #
# Generating Poisson Inputs #
# ************************* #

def poisson_input(lamda = 5, dim: int = 2, length: int = 100) -> dict[int, np.ndarray]:
    """
    Generate a random Poisson input stream of given lambda, dimension and length.
    https://neuronaldynamics.epfl.ch/online/Ch7.S3.html
    """
    inputs = np.zeros(shape=(dim, length))

    for d in range(dim):
        isi = np.random.poisson(lam=lamda, size=length) # inter-spike interval
        isi = [ [0]*i + [1] for i in isi ]
        isi = np.array([t for interval in isi for t in interval])[:length]
        inputs[d, :] = isi

    inputs = {t: inputs[:,t] for t in range(length)}

    return inputs

# ************************************ #
# Retrieving Head and Tail of an Input #
# ************************************ #

def head(U: dict[int, np.ndarray], size: int) -> dict[int, np.ndarray]:
    """Retrieve the first size elements of input U."""
    return {i: U[i] for i in range(size)}

def tail(U: dict[int, np.ndarray], size: int) -> dict[int, np.ndarray]:
    """Retrieve the last size elements of input U."""
    max_index = max(U.keys())
    return {i: U[max_index - size + i + 1] for i in range(size)}

# ******************************* #
# Concatenating Two Input Streams #
# ******************************* #

def concatenate(U1: dict[int, np.ndarray], U2: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Concatenate two input streams."""
    combined_stream = {**U1}
    offset = max(U1.keys()) + 1
    combined_stream.update({i + offset: U2[i] for i in U2})
    return combined_stream

# ************************************** #
# Generating Inputs with Trigger Pattern #
# ************************************** #

def generate_ticks(count=20, pattern_length=20, input_length=1000):
    """
    Generate K=count ticks between 0 and the N=input_length 
    separated by at least L=pattern_length time steps. 
    The function is problematic in case of impossible sampling...
    """
    ticks = []
    
    # avoid tick generation if sequence is too short
    flag = (input_length > 1.25 * count * pattern_length) # 1.25 is arbitrary...

    while flag and len(ticks) < count:
        
        # 5 warming time steps minimum
        candidate = np.random.randint(low=5, high=input_length, size=(1))
        candidate = int(candidate)
        
        if (candidate + pattern_length <= input_length) and \
            all(abs(candidate - num) > pattern_length for num in ticks):

            ticks.append(candidate)
    
    ticks = sorted(ticks)

    return ticks

def insert(pattern: dict[int, np.ndarray], stream: dict[int, np.ndarray], start: int) -> dict[int, np.ndarray]:
    """Insert pattern into stream at specified start time, overwriting existing values."""

    if start >= len(stream):
        return False
    
    for i in range(len(pattern)):
        stream[start + i] = pattern[i]

    return stream

def mix_input(stream: dict[int, np.ndarray], pattern: dict[int, np.ndarray], tics: list[int]) -> dict[int, np.ndarray]:
    """Insert pattern at specified time steps in stream."""

    for t in tics:
        insert(pattern, stream, t)

    return stream

def mix_input2(stream: dict[int, np.ndarray], pattern: dict[int, np.ndarray], interval: int) -> dict[int, np.ndarray]:
    """Insert pattern at regular intervals in stream."""

    pattern_positions = [k * len(stream) // (interval + 1) for k in range(1, interval + 1)]

    for idx, position in enumerate(pattern_positions):
        insert(pattern, stream, position + idx * len(pattern))

    return stream

def find_pattern(pattern: dict[int, np.ndarray], stream: dict[int, np.ndarray]) -> list[int]:
    """Find occurrences of a pattern in a stream and return starting positions."""

    occurrences = []

    for t in range(len(stream) - len(pattern) + 1):
        if all(np.array_equal(stream[t + i], pattern[i]) for i in range(len(pattern))):
            occurrences.append(t)

    return occurrences

def generate_input(input_dim=1, input_length=300, mode="random", lamda=5, triggers=True, nb_triggers=10, trigger_length=10):
    """
    Generates a random input stream interspersed with binary trigger patterns.

    Args:
        input_dim (int): Dimension of each element in the input stream.
        input_length (int): Length of the input stream.
        nb_triggers (int): Number of trigger patterns to insert.
        trigger_length (int): Length of each trigger pattern.

    Returns:
        tuple: A tuple containing the generated input stream and the list of indices
               where the trigger patterns end.
    """

    # Generate input stream and insert trigger pattern
    if mode == "random":
        U = random_input(dim=input_dim, length=input_length)
    elif mode == "poisson":
        U = poisson_input(lamda=lamda, dim=input_dim, length=input_length)

    if triggers:
        pattern = {i: np.random.randint(2, size=input_dim) for i in range(trigger_length)}
        triggers = generate_ticks(nb_triggers, trigger_length, len(U))
        U = mix_input(U, pattern, triggers)

    return U, triggers

# ******************** #
# Generating a Network #
# ******************** #

def generate_network(nb_inputs: int, nb_nodes: int, nb_input_connections: int = 5, nb_internal_connections: int = 30) -> tuple[list[int], list[int], list[tuple[tuple[int, int], float]]]:
    """Generate a random Boolean network with input and internal connections."""
    input_nodes = list(range(nb_inputs))
    internal_nodes = list(range(nb_inputs, nb_inputs + nb_nodes))
    edges = []

    def create_edges(nb_connections: int, src_range: tuple[int, int], dest_range: tuple[int, int]):
        """Helper to add edges between nodes in specified ranges."""
        count = 0
        while count < nb_connections:
            src = np.random.randint(*src_range)
            dest = np.random.randint(*dest_range)
            if (src, dest) not in [e[0] for e in edges]:  
                weight = np.random.normal(np.random.choice([-1.5, 1.5]), 0.5)
                edges.append(((src, dest), weight))
                count += 1

    create_edges(nb_input_connections, (0, nb_inputs), (nb_inputs, nb_inputs + nb_nodes))
    create_edges(nb_internal_connections, (nb_inputs, nb_inputs + nb_nodes), (nb_inputs, nb_inputs + nb_nodes))

    return input_nodes, internal_nodes, edges


# *********************** #
# Example Usage and Tests #
# *********************** #

if __name__ == "__main__":
    
    pass
    # # Test distort
    # A = np.random.normal(size=(5,5))
    # mask = np.abs(A) > 0.8
    # A = A * mask
    # A_d = distort(A, noise=10.)
    # print("A", A)
    # print("A distorted", A_d)
    # print("Check same signs", A * A_d >= 0)

    # # Test inputs
    # U, ticks = generate_input(input_dim=1, 
    #                           input_length=1000, 
    #                           mode="poisson", 
    #                           lamda=3, 
    #                           triggers=True, 
    #                           nb_triggers=20, 
    #                           trigger_length=20)
    # print(ticks)
    # inputs = [int(x.item()) for x in list(U.values()) ]
    # print("Input stream:\n", inputs)
    # for t in ticks:
    #     print(t)
    #     print(inputs[t:t+20])

    # Test attractors
    # from attractors import * # XXX import problem

    # N = generate_network(1, 10, 5, 30)
    # # print(N[0])
    # # print(N[1])
    # # print(N[2])
    # # print(len(N[2]))
    # A = net_to_aut(N)
    # dico_cycles = simple_cycles(A)
    # for k, v in dico_cycles.items():
    #     print(k)
    #     print(v, "\n")
    # SCC = largest_list(list(dico_cycles.keys()))
    # #SCC = find_SCC_0(list(dico_cycles.keys()))
    # (C, n) = dico_cycles[SCC]
    # print(C,n)

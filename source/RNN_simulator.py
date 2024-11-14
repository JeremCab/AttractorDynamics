# -*- coding: utf-8 -*-

import numpy as np

# **************** #
# GLOBAL NOTATIONS #
# **************** #

# A: weight matrix		internal connections
#						A[i,j] = w iff x_i --w--> x_j

# B1: weight matrix		input connections (from input to internal cells)
# 						B1[i,j] = w iff u_i --w--> x_j

# B2: weight matrix		interactive connections (from internal to input cells)
#						B2[i,j] = w iff x_i--w--> u_j

# b: bias vector		bias connections (from env. to internal cells)
#						b[i] = w iff env--w--> x_i

# x: initial state      initial activation values of internal cells
#						x[i] = a iff x_i(0) = a

# U: input dictionary	keys: time steps; values: input vectors
#						U[t] = input vector at time step t
#						if time step not specified, input vector = [0,...,0]

# ******************** #
# ACTIVATION FUNCTIONS #
# ******************** #

def theta(x, threshold=1):
    """Hard-threshold activation function."""
    return 1 if x >= threshold else 0

theta = np.vectorize(theta, otypes=[np.ndarray])

def sigma(x):
    """Linear-sigmoidal activation function."""
    if x < 0:
        return 0
    elif x < 1:
        return x
    return 1

sigma = np.vectorize(sigma, otypes=[np.ndarray])

# ********* #
# STDP Rule #
# ********* #

def STDP(A, x_tminus1, x_t, A_init, eta=0.01, plumb=1, bounds=(-0.5, 0.5)):
    """
    Applies a targeted STDP (Spike-Timing Dependent Plasticity) rule to non-interactive connections in matrix A.
    
    Args:
        A (np.ndarray): Current weight matrix.
        x_tminus1 (np.ndarray): State vector at time t-1.
        x_t (np.ndarray): State vector at time t.
        A_init (np.ndarray): Default (initial) weight matrix for bounding weight changes.
        eta (float, optional): Learning rate for STDP. Defaults to 0.01.
        plumb (float, optional): Coefficient favoring weight decrease. Defaults to 1.
        bounds (tuple, optional): Tuple of lower and upper bounds for weight changes. Defaults to (-0.5, 0.5).

    Returns:
        np.ndarray: Updated weight matrix A.
    """
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[j,i] != 0:  # Only apply STDP to non-zero connections
                # Introduce noise to learning rate by modifying eta slightly
                factor = (1.05 - 0.95) * np.random.random_sample() + 0.95
                eta_modified = eta * factor
                # Calculate the new weight with STDP adjustments
                temp_weight = A[j,i] + eta_modified * (x_t[i].item() * x_tminus1[j].item() - plumb * x_tminus1[i].item() * x_t[j].item())

                # Clip the weight within bounds defined by A_init and bounds parameter
                A[j,i] = np.clip(temp_weight, A_init[j,i] + bounds[0], A_init[j,i] + bounds[1])

                if A[j][i] == 0:
                    print("Connection dead...")

    return A

# ******** #
# GSP Rule #
# ******** #

def StochasticSynapses(A_init, mu_A, sigma2_A, bounds=(-0.5, 0.5)):
    """
    Gaussian sampling of (non-interactive) synaptic weights.
    Each non-zero weight a_ij of A is sampled in the normal distribution N(mu_ij, sigma2_ij)
    If A stretches beyond the bounds [A_init + bounds[0], A_init + bounds[0]], then it is clipped.
    
    Args:
        A_init (np.ndarray): Default (initial) weight matrix for bounding weight changes.
        mu_A (np.ndarray): Matrix of means w.r.t. which the new weights will be sampled.
        sigma2_A (np.ndarray): Matrix of variances w.r.t. which the new weights will be sampled.
        bounds (tuple, optional): Tuple of lower and upper bounds for weight changes. Defaults to (-0.5, 0.5).

    Returns:
        np.ndarray: Updated weight matrix A.
    """

    mask = A_init != 0
    dim = A_init.shape[0], A_init.shape[1]
    A = np.random.normal(mu_A, sigma2_A, size=dim)
    A = A * mask
    A = np.maximum(A, A_init + bounds[0])  # clipping values (adding negative bound)
    A = np.minimum(A, A_init + bounds[1])  # clipping values (adding positive bound)
    
    return A_init, A, mu_A, sigma2_A

def GSP(A_init, A_minus1, A, mu_A, sigma2_A, eta=0.1, reward=0, bounds=(-0.5, 0.5)):
    """
    Applies a global stochastic plasticity rule to (non-interactive) connections in matrix A.
    if the reward is positive, then the mean is shifted towards tthe direction of A_minus1 -> A
    and the std is decreased. Then, a new weight matrix A is samples according to this new mean and std.
    Otherwise, A remains unchanged.
    
    Args:
        A_init (np.ndarray): Default (initial) weight matrix for bounding weight changes.
        A (np.ndarray): Weight matrix at current time step.
        A_minus1 (np.ndarray): Weight matrix at previous time step.
        mu_A (np.ndarray): Current matrix of means w.r.t. which the new weights have been sampled.
        sigma2_A (np.ndarray): Current matrix of variances w.r.t. which the new weights have been be sampled.
        reward (float): Postive learning rate.
        reward (float): Postive reward.
        bounds (tuple, optional): Tuple of lower and upper bounds for weight changes. Defaults to (-0.5, 0.5).

    Returns:
        list: List of original weights, updated weights, updated means, and updated stds.
    """
        
    if reward > 0:

        A_diff_sign = ((A_minus1 - A) > 0) * 2 - 1

        # XXX this rule can be more sophisticated
        mu_A = mu_A + (eta * A_diff_sign)                    # update mean (shifting)
        sigma2_A = np.maximum(sigma2_A - eta * sigma2_A, 0)  # update std (decreasing)

        A_init, A, mu_A, sigma2_A = StochasticSynapses(A_init, mu_A, sigma2_A, bounds=(-0.5, 0.5))

    return [A_init, A, mu_A, sigma2_A]

# ********* #
# Simulator #
# ********* #

def simulation(A, B1, B2, b, x, U, epoch=100, stdp="off"):
    """
    Runs a simulation on a network defined by matrices A, B1, B2, b, and input stream U.
    
    Args:
        A (np.ndarray): Weight matrix for internal states.
        B1 (np.ndarray): Weight matrix for influenc e of inputs on internal states.
        B2 (np.ndarray): Weight matrix for influence of internal states on inputs.
        b (np.ndarray): Bias vector for internal state update.
        x (np.ndarray): Initial state vector.
        U (dict): Input dictionary with keys as time steps and values as input vectors.
        epoch (int, optional): Number of simulation steps. Default to 100.
        stdp (str or list, optional): STDP parameters as [A_init, eta, plumb, bounds] or "off" to disable STDP.

    Returns:
        list: Contains the history of states and the final matrices (A, B1, B2, b).
    """
    dim = B1.shape[0] + x.shape[0]                        # Calculate state space dimension
    history = np.zeros([dim, epoch])                      # Initialize history with dummy states
    synapses = np.zeros([x.shape[0], x.shape[0], epoch])  # Initialize synapses with dummy states
    synapses[:, :, 0] = A
    
    for i in range(epoch):
        # Retrieve or initialize input for current time step
        u = U.get(i, np.zeros([B1.shape[0], 1]))  
        u = theta(np.dot(B2.T, x) + u)             # Processed input after interactive signal
        
        # Compute the new internal state
        x_plus = theta(np.dot(A.T, x) + np.dot(B1.T, u) + b)
        history[:, i] = np.vstack([u, x]).reshape(-1)

        # Apply STDP if enabled
        if stdp != "off":
            A = STDP(A, x, x_plus, stdp[0], stdp[1], stdp[2], stdp[3])
        synapses[:, :, i] = A

        x = x_plus  # Update state for the next iteration
    
    # Remove the initial dummy state and return the state history and final matrices
    return history, synapses, [np.copy(A), np.copy(B1), np.copy(B2), np.copy(b), x]


# *********************** #
# Example Usage and Tests #
# *********************** #

if __name__ == "__main__":

    # # Run STDP for multiple steps to observe weight updates
    # A = np.array([[0, 0.5, 0, 0, 0], [0, 0.5, 0, 0, 2.7], [0, 0.5, 0, 0, 0], [0, 0, -0.5, 0, 0], [0, 0, 0, 0, 0]])
    # A_init = np.copy(A)  # Immutable copy of A for bounding in STDP
    # x_tminus1 = np.array([[1], [0], [1], [0], [1]])
    # x_t = np.array([[0], [1], [0], [1], [0]])

    # for i in range(60):
    #     A = STDP(A, x_tminus1, x_t, A_init)
    #     print("Step", i, "\n", A)

    # Run simulation with STDP enabled
    A_initial = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    B1 = np.array([[1., 0., 0.]])
    B2 = np.array([[0], [0], [0]])
    b = np.array([[0.], [0.], [0.]])
    x = np.zeros([3, 1])
    # U = random_input(dim=2, length=100)
    U = {t: np.random.randint(2, size=(1, 1)) for t in range(1)}

    history, synapses, matrices = simulation(A_initial, B1, B2, b, x, U, epoch=20, stdp=[A_initial, 0.1, 1, (-0.5, 0.5)])
    print("********* U *********\n", U)
    print("********* history *********\n", history)
    print("********* A_initial *********\n", A_initial)
    print("********* synapses *********\n")
    for i in range(20):
        print(synapses[:, :, i])
    print("********* A *********\n", matrices[0])
    print("********* B1 *********\n", matrices[1])
    print("********* B2 *********\n", matrices[2])
    print("********* b *********\n", matrices[3])
    print("********* x *********\n", matrices[4])

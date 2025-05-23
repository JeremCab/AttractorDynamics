import subprocess
import itertools
from tqdm import tqdm
import os


# set current dir to script folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_simulation(mode, input_length, trigger_length, nb_triggers,  seed, 
                   temperature, cooling_rate, eta, plumb, bounds, noise, network):
    """
    Runs the `simul_stdpgp.py` script with the specified parameters.
    """
    
    # Construct the command
    cmd = [
        "python", "simul_stdpgp.py",
        "--mode", mode,
        "--seed", str(seed),
        "--input_length", str(input_length),
        "--trigger_length", str(trigger_length),
        "--nb_triggers", str(nb_triggers),
        "--temperature", str(temperature),
        "--cooling_rate", str(cooling_rate),
        "--eta", str(eta),
        "--plumb", str(plumb),
        "--bounds", bounds, # bounds as string
        "--noise", str(noise),
        "--network", str(network)
    ]

    # Run the command
    print(f"Running simulation with: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print results
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Simulation failed with return code {result.returncode}")
    else:
        print("Simulation completed successfully.")

# JE REFAIS LES EXPéRIENCES POUT ETA = 0.1

def run_grid_search():
    """
    Run a grid search over different parameter combinations.
    """

    # Define parameter grids
    modes = ["stdp", "gp", "stdp-gp"]                 # 3 modes "stdp", "gp", "stdp-gp"
    input_lengths = [1001]                            # fixed
    trigger_lengths = [50]                            # fixed
    nb_triggers = [0, 1, 3, 5, 7, 9] #[0, 1, 3, 5, 7, 9, 11] #         # several triggers
    seeds = [42, 79, 82, 83, 47, 49, 13, 77, 55, 15]  # 10 seeds
    temperatures = [10.0]                             # fixed
    cooling_rates = [0.995]                           # fixed
    etas = [0.1] #, 0.01, 0.001, 0.0001] # [0.025]            # 4 etas # XXX XXX XXX
    plumbs = [1.0]                                    # fixed
    bounds = ["minus0.4999,1.4999"] #["minus0.4999,0.4999"]           # fixed
    noises = [0.25]                                   # fixed
    network = ["bgt"] # ["random"]                    # fixed

    # Generate all combinations of parameters using itertools.product
    param_grid = itertools.product(modes, input_lengths, trigger_lengths, nb_triggers,
                                   seeds, temperatures, cooling_rates, etas,
                                   plumbs, bounds, noises, network)

    # Loop over each combination of parameters and run the simulation
    for params in tqdm(param_grid):
        (mode, input_length, trigger_length, nb_triggers, seed, temperature, 
         cooling_rate, eta, plumb, bounds, noise, network) = params

        # Run the simulation for this parameter combination
        run_simulation(mode, input_length, trigger_length, nb_triggers, seed, 
                       temperature, cooling_rate, eta, plumb, bounds, noise, network)


if __name__ == "__main__":
    # Run the grid search
    run_grid_search()


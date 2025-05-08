import subprocess
import itertools
from tqdm import tqdm
import os


# set current dir to script folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_simulation(mode, input_length, trigger_length, nb_triggers,  seed, 
                   temperature, cooling_rate, eta, plumb, bounds, noise):
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
        "--noise", str(noise)
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


def run_grid_search():
    """
    Run a grid search over different parameter combinations.
    """

    # Define parameter grids
    modes = ["stdp-gp"]                               # 1 mode only
    input_lengths = [1001]                            # fixed
    trigger_lengths = [50]                            # fixed
    nb_triggers = [10]                                # fixed
    seeds = [77]                                      # 1 seed only
    temperatures = [10.0]                             # fixed
    cooling_rates = [0.995]                           # fixed
    etas = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030] # various etas
    plumbs = [1.0]                                    # fixed
    bounds = ["minus0.4999,1.4999"]                   # fixed
    noises = [0.3]                                    # fixed

    # Generate all combinations of parameters using itertools.product
    param_grid = itertools.product(modes, input_lengths, trigger_lengths, nb_triggers,
                                   seeds, temperatures, cooling_rates, etas,
                                   plumbs, bounds, noises)

    # Loop over each combination of parameters and run the simulation
    for params in tqdm(param_grid):
        (mode, input_length, trigger_length, nb_triggers, seed, temperature, 
         cooling_rate, eta, plumb, bounds, noise) = params

        # Run the simulation for this parameter combination
        run_simulation(mode, input_length, trigger_length, nb_triggers, seed, 
                       temperature, cooling_rate, eta, plumb, bounds, noise)


if __name__ == "__main__":
    # Run the grid search
    run_grid_search()


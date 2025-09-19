#!/usr/bin/env python3
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MOLECULES = ["chignolin", "trpcage", "chignolin_implicit", "trpcage_implicit"]
METHODS = ["ours", "rmsd", "tae", "vde", "deeptda", "deeptica"]
K_VALUES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
GPU_MAPPING = {100: 0, 200: 1, 500: 2, 1000: 3, 2000: 4, 5000: 5, 10000: 6, 20000: 7}

def run_experiment(molecule, method, k, gpu_id=None):
    """Run a single experiment on specified GPU."""
    gpu_info = f" on GPU {gpu_id}" if gpu_id is not None else ""
    print(f"Running experiment: {molecule}/{method}/{k}{gpu_info}")
    
    cmd = [
        "python", "src/smd.py", 
        "--config-path", f"config/{molecule}",
        "--config-name", method,
        f"simulation.k={k}",
        "simulation.timestep=2",
        "simulation.num_steps=100000",
    ]
    
    # Set GPU environment variable
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/kiyoung/tlc", env=env)
        if result.returncode != 0:
            print(f"Error in {molecule}/{method}/{k}{gpu_info}: {result.stderr}")
            return False
        print(f"✓ Completed: {molecule}/{method}/{k}{gpu_info}")
        return True
    except Exception as e:
        print(f"Exception in {molecule}/{method}/{k}{gpu_info}: {e}")
        return False

def run_parallel_experiments(molecule, method):
    """Run experiments for all K values in parallel on different GPUs."""
    print(f"\n=== Running parallel experiments for {molecule}/{method} ===")
    
    # Create futures for each K value
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for k in K_VALUES:
            gpu_id = GPU_MAPPING[k]
            future = executor.submit(run_experiment, molecule, method, k, gpu_id)
            futures[future] = (k, gpu_id)
        
        # Wait for all experiments to complete
        results = {}
        for future in as_completed(futures):
            k, gpu_id = futures[future]
            try:
                success = future.result()
                results[k] = success
                print(f"✓ {molecule}/{method}/{k} on GPU {gpu_id}: {'Success' if success else 'Failed'}")
            except Exception as e:
                print(f"✗ {molecule}/{method}/{k} on GPU {gpu_id}: Exception {e}")
                results[k] = False
    
    return results

def run_experiments():    
    for molecule in MOLECULES:
        for method in METHODS:
            run_parallel_experiments(molecule, method)

def main():
    """Main function to run experiments and generate tables."""
    print("Starting comprehensive experiments and analysis...")
    print("Note: Using parallel GPU execution (K=100->GPU0, K=200->GPU1, K=500->GPU2, K=1000->GPU3, K=2000->GPU4, K=5000->GPU5, K=10000->GPU6, K=20000->GPU7)")
    print("Each molecule/method combination will run all K values in parallel")
    run_experiments()

if __name__ == "__main__":
    main()

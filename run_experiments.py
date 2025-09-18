#!/usr/bin/env python3
import os
import json
import subprocess
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MOLECULES = ["chignolin", "trpcage"]
METHODS = ["ours", "rmsd", "tae", "vae", "deeptda", "deeptica"]
# K_VALUES = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
K_VALUES = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
GPU_MAPPING = {1000: 0, 2000: 1, 5000: 2, 10000: 4, 20000: 5, 50000: 6, 100000: 7}  # K value -> GPU ID

def run_experiment(molecule, method, k, gpu_id=None):
    """Run a single experiment on specified GPU."""
    gpu_info = f" on GPU {gpu_id}" if gpu_id is not None else ""
    print(f"Running experiment: {molecule}/{method}/{k}{gpu_info}")
    
    cmd = [
        "python", "src/smd.py", 
        "--config-path", f"config/{molecule}",
        "--config-name", method,
        f"simulation.k={k}",
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

def extract_metrics(molecule, method, k):
    """Extract metrics from saved JSON file."""
    metrics_file = f"/home/kiyoung/tlc/res/{molecule}/{method}/{k}/metrics.json"
    if not os.path.exists(metrics_file):
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics for {molecule}/{method}/{k}: {e}")
        return None

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

def collect_all_metrics():
    """Collect metrics for all experiments with parallel GPU execution."""
    all_metrics = {}
    
    for molecule in MOLECULES:
        for method in METHODS:
            print(f"\nProcessing {molecule}/{method}...")
            
            # Check existing metrics first
            missing_experiments = []
            for k in K_VALUES:
                metrics = extract_metrics(molecule, method, k)
                if metrics:
                    all_metrics[(molecule, method, k)] = metrics
                    print(f"✓ Found existing metrics for {molecule}/{method}/{k}")
                else:
                    missing_experiments.append(k)
            
            # Run missing experiments in parallel
            if missing_experiments:
                print(f"Missing experiments for {missing_experiments}, running in parallel...")
                run_parallel_experiments(molecule, method)
                
                # Collect metrics after running experiments
                for k in missing_experiments:
                    metrics = extract_metrics(molecule, method, k)
                    if metrics:
                        all_metrics[(molecule, method, k)] = metrics
    
    return all_metrics

def generate_all_tables(all_metrics):
    """Generate all LaTeX tables in a single file with K values as rows."""
    all_latex_lines = []
    
    for molecule in MOLECULES:
        all_latex_lines.append(f"\n% Results for {molecule.capitalize()}")
        for method in METHODS:
            table_lines = generate_method_table_lines(molecule, method, all_metrics)
            all_latex_lines.extend(table_lines)
            all_latex_lines.append("")  # Empty line between tables
    
    # Write all tables to a single file
    output_file = "/home/kiyoung/tlc/tables/all_results_tables.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(all_latex_lines))
    print(f"All LaTeX tables saved to: {output_file}")
    print(f"Generated {len(MOLECULES) * len(METHODS)} tables in single file")

def generate_method_table_lines(molecule, method, all_metrics):
    """Generate LaTeX table lines for a specific molecule/method combination with K values as rows."""
    latex_lines = [
        "\\begin{table}[h]", "\\centering", 
        f"\\caption{{Results for {molecule.capitalize()} using {method.upper()}}}",
        "\\begin{tabular}{l|c|c|c}", "\\hline",
        "K & RMSD ($\\pm$ std) & THP (\\%) & ETS ($\\pm$ std) \\\\", "\\hline"
    ]
    
    for k in K_VALUES:
        metrics = all_metrics.get((molecule, method, k))
        if metrics:
            rmsd = f"{metrics['rmsd']:.2f}" + (f" $\\pm$ {metrics['rmsd_std']:.2f}" if 'rmsd_std' in metrics else "")
            thp = f"{metrics.get('thp', 0):.1f}"
            ets = f"{metrics['ets']:.2f}" + (f" $\\pm$ {metrics['ets_std']:.2f}" if 'ets_std' in metrics else "") if 'ets' in metrics else "N/A"
            latex_lines.append(f"{k} & {rmsd} & {thp} & {ets} \\\\")
        else:
            latex_lines.append(f"{k} & N/A & N/A & N/A \\\\")
    
    latex_lines.extend(["\\hline", "\\end{tabular}", f"\\label{{tab:{molecule}_{method}_results}}", "\\end{table}"])
    
    return latex_lines

def plot_rmsd_ets_tradeoff(metrics_file_path="/home/kiyoung/tlc/all_metrics.json", output_dir="/home/kiyoung/tlc/figures"):
    """
    Plot RMSD vs ETS trade-off for all methods and K values.
    Creates separate figures for each molecule (chignolin, trpcage).
    
    Args:
        metrics_file_path: Path to the JSON file containing all metrics
        output_dir: Directory to save the plots
    """
    # Load metrics data
    if not os.path.exists(metrics_file_path):
        print(f"Metrics file not found: {metrics_file_path}")
        return
    
    with open(metrics_file_path, 'r') as f:
        all_metrics = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use global configuration variables
    molecules = MOLECULES
    methods = METHODS
    k_values = K_VALUES
    
    # Color and marker settings for each method
    method_styles = {
        "ours": {"color": "red", "marker": "o", "label": "Ours"},
        "rmsd": {"color": "blue", "marker": "s", "label": "RMSD"},
        "tae": {"color": "green", "marker": "^", "label": "TAE"},
        "vae": {"color": "orange", "marker": "v", "label": "VAE"},
        "deeptda": {"color": "purple", "marker": "d", "label": "DeepTDA"},
        "deeptica": {"color": "brown", "marker": "p", "label": "DeepTICA"}
    }
    
    # Create plots for each molecule
    for molecule in molecules:
        plt.figure(figsize=(4, 4))
        
        # Plot data for each method
        for method in methods:
            rmsd_values = []
            ets_values = []
            k_labels = []
            
            # Extract RMSD and ETS values for each K
            for k in k_values:
                key = f"{molecule}_{method}_{k}"
                if key in all_metrics and all_metrics[key]:
                    metrics = all_metrics[key]
                    if 'rmsd' in metrics and 'ets' in metrics:
                        rmsd_values.append(metrics['rmsd'])
                        ets_values.append(metrics['ets'])
                        k_labels.append(k)
            
            # Plot if we have data
            if rmsd_values and ets_values:
                style = method_styles[method]
                plt.plot(rmsd_values, ets_values, 
                           color=style["color"], 
                           marker=style["marker"], 
                           linewidth=2, 
                           alpha=0.7,
                           label=style["label"])
                
                # Add K value annotations
                # for i, (rmsd, ets, k) in enumerate(zip(rmsd_values, ets_values, k_labels)):
                #     plt.annotate(f'{k//1000}k', 
                #                (rmsd, ets), 
                #                xytext=(5, 5), 
                #                textcoords='offset points', 
                #                fontsize=8, 
                #                alpha=0.8)
        
        # Customize the plot
        plt.xlabel('RMSD', fontsize=14)
        plt.ylabel('ETS', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(output_dir, f'{molecule}_rmsd_ets_tradeoff.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot for {molecule}: {output_file}")
        
        # Also save as PDF
        output_file_pdf = os.path.join(output_dir, f'{molecule}_rmsd_ets_tradeoff.pdf')
        plt.savefig(output_file_pdf, bbox_inches='tight')
        print(f"Saved plot for {molecule}: {output_file_pdf}")
        
        plt.close()
    
    print(f"All RMSD vs ETS trade-off plots saved to: {output_dir}")

def main():
    """Main function to run experiments and generate tables."""
    print("Starting comprehensive experiments and analysis...")
    print("Note: Using parallel GPU execution (K=1000->GPU0, K=2000->GPU1, K=5000->GPU2, K=10000->GPU4, K=20000->GPU5, K=50000->GPU6, K=100000->GPU7)")
    print("Each molecule/method combination will run all K values in parallel")
    
    all_metrics = collect_all_metrics()
    print(f"Collected metrics for {len(all_metrics)} experiments")
    
    # Save all metrics
    with open("/home/kiyoung/tlc/all_metrics.json", 'w') as f:
        json.dump({f"{k[0]}_{k[1]}_{k[2]}": v for k, v in all_metrics.items()}, f, indent=2)
    
    # Generate tables for each molecule/method combination
    generate_all_tables(all_metrics)
    
    # Generate RMSD vs ETS trade-off plots
    print("\nGenerating RMSD vs ETS trade-off plots...")
    plot_rmsd_ets_tradeoff()
    
    print("Analysis completed! All LaTeX tables and RMSD vs ETS plots saved.")

if __name__ == "__main__":
    main()

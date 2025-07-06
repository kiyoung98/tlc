import argparse

def init_dataset_args():
    parser = argparse.ArgumentParser(description="Dataset builder for Contrasstive learning")

    parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="alanine")
    parser.add_argument("--temperature", type=float, help="Temperature to use", default=600.0)
    parser.add_argument('--traj_dir', action='append', help='List of trajectories to make a dataset', required=True)
    parser.add_argument("--data_per_traj", type=int, help="Dataset size", default=10000)
    parser.add_argument("--dataset_version", type=str, help="Dataset version", default="debug-v1")
    parser.add_argument("--time_lag", type=int, help="Time lag", default=3)
    # parser.add_argument("--positive_sample_augmentation", type=int, help="Positive sample time index augmentation", default=100)
    # parser.add_argument("--negative_sample_augmentation", type=int, help="Negative sample time index augmentation", default=100_000)

    args = parser.parse_args()
    
    return args
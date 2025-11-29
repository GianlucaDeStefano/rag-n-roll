"""
    This script lets you optimize a dataset using the selected algorithm.
"""

import argparse
from src.attacks import optimize_dataset


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset of contexts to optimize')
    parser.add_argument('--opt_name', type=str, help='Name of the optimization algorithm to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--injection_strategy', type=str, default=None, help='Some attacks need an injection strategy, specify it here')
    parser.add_argument('--mode', type=str, default=None, help='Mode of the attack')
    parser.add_argument('--regularize', action='store_true',default=False, help='Use regularize to decrease perplexity')

    args = parser.parse_args()
    
    # Optimize the dataset
    attack_args = {k:v for k,v in vars(args).items() if k not in ['dataset_path','opt_name']}
    optimize_dataset(args.dataset_path, args.opt_name, **attack_args)
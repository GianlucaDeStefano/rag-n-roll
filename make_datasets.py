"""
    This script is used to generate a dataset for the benchmarking process starting from existing resources.
    The dataset is saved in a json file and can be used to run the bramchmarking process.
    The script accepts the following arguments:
        - dataset_name: name of the dataset to generate. Currently we support ['NQopen'].
        - n_queries: number of queries to generate (Default: None)
        - n_distractor_docs: number of distractor documents to generate (Default: None)
        - generate_independent_queries: boolean indicating if the queries are generated independently from the documents (Default: None)
        - output_path: path where to save the dataset, if 
"""
import argparse
import os
from src.datasets import make_dataset

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--output_name', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default='data/datasets')
    parser.add_argument('--n_queries', type=int, default=None)
    parser.add_argument('--n_distractor_docs', type=int, default=None)
    parser.add_argument('--generate_independent_queries', type=bool, default=True)
    args = parser.parse_args()
    
    if args.output_name is None:
        args.output_name = args.dataset_name
            
    output_folder = f'{args.output_folder}/{args.output_name.lower()}'
    #assert not os.path.exists(output_folder), f"Output folder {output_folder} already exists"
    os.makedirs(output_folder,exist_ok=True)
    args.output_folder = output_folder
    
    # Construct the dataset
    dataset = make_dataset(**vars(args))
    
    # Save the constructed dataset to the output folder
    dataset.save(output_folder)
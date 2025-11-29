from datetime import datetime
import os
import subprocess
import logging

FOLDER = os.path.dirname(__file__)
AVAILABLE_ATTACKS = [f for f  in os.listdir(FOLDER) if os.path.isdir(os.path.join(FOLDER, f)) and not f.startswith('.') and not f.startswith('__')]

DEFAULT_DATASET_MNT_FOLDER = '/dataset'

def optimize_dataset(dataset_path,opt_name,gpu=0, **kwargs): 
    """
    Optimize the dataset using the specified optimization method
    Args:
        dataset_path (str): Path to the dataset to optimize
        opt_name (str): Name of the optimization method to use 
    """
    
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
    assert os.path.isdir(dataset_path), f"Dataset path {dataset_path} is not a folder"
    assert opt_name in AVAILABLE_ATTACKS, f"Attack {opt_name} not found. Available attacks are: {AVAILABLE_ATTACKS}"
    
    # Folder containing the optimization method
    opt_folder = os.path.join(FOLDER, opt_name)
    
    # Build the docker image to use to run the optimization
    logging.info(f"Building docker image for optimization {opt_name}")
    subprocess.run(['docker','build', '-t', opt_name, f"{opt_folder}/docker"])
    
    # Construct the command to run the optimization
    command = f'python optimize_dataset.py --dataset_path={DEFAULT_DATASET_MNT_FOLDER}'
    for k,w in kwargs.items():
        
        if w:
            command += f' --{k} {w}'
    
    # Run the optimization 
    start_time = datetime.now()
    full_docker_command = " ".join(['docker','run','-v',f'{opt_folder}:/home','-v',f'{dataset_path}:{DEFAULT_DATASET_MNT_FOLDER}','--gpus',f'device={gpu}','-v','~/.cache/huggingface:/.cache/huggingface','-e','HF_HOME=/.cache/huggingface','--rm', opt_name,command])
    logging.info(f"Running optimization {opt_name} on dataset {dataset_path}, command: {full_docker_command} start time: {start_time}")
    
    os.system(full_docker_command)

import gc
import random
import pandas as pd
import numpy as np
import torch
from pynvml import *


def infer_device():
    device = 'cpu'
    
    if torch.backends.mps.is_available():
        device = 'mps'
    
    if torch.cuda.is_available():
        device = 'cuda'
    
    return device

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    os.environ['RANDOM_SEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    
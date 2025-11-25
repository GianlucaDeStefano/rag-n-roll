# Attacks
Rag-n'-roll implements data re-ranking/retrieval/
corruption attacks from the literature and uses them to optimize the adversarial documents in a benchmark. 

## Add an attack
The implementation of each attack is in a separate directory in the `src/attacks` root folder. 

Each directory can contain at least the following files:
- `optimize_dataset.py`: the implementation of the attack used to optimize the adversarial documents. 
This script has a mandatory parameter --dataset_path that specifies the path to the dataset to be optimized.
All other parameters are optional and can be used to configure the attack.
- `dockerfile`: the dockerfile used to build the docker image for the attack.
- `requirements.txt`: the requirements file for the attack.
- `install.sh`: the installation script for the attack to download models and other necessary data.
- `README.md`: the documentation of the attack. 

## Optimize a dataset
To optimize a dataset using an attack, you can use the following command:
```bash
python src/attacks/<attack_name>/optimize_dataset.py --benchmark_path <path_to_benchmark>
```
This script will optimize the dataset at the specified path using the attack implementation in the `src/attacks/<attack_name>` directory. The results will be a set of optimized documents that will be saved in `<path_to_benchmark>/optimized/<attack_name>.jsonl`
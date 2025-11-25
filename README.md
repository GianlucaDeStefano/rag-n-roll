<p align="center">
	<a href="">
		<img align="center" alt="Rag and Roll" src="assets/logo.jpg" height="195">
	</a>
</p>
<p align="center">
	<span><b> Rag and Roll </b></span>
</p>

### Installation
1) Create a conda environment with python 3.10
```bash
conda create -n rag-n-roll python=3.10
conda activate rag-n-roll
```

2) Install the requirements
```bash
pip install -r docker/requirements.txt
```

3) Add environment variables to .env file to be able to use closed models such as GPT4.

4) Make sure that docker is running on the machine as we will need it to run the optimization processes. 

### Download our sataset
You can download our NQopen [here](https://mega.nz/file/pb0nESiI#TFtiv_K4V6KyVPfJs9J8T55m5w7wEYj3yrOyEr3DxpQ)
Place the unzipped datasets in the `data/benchmarks` directory.

### Rebuilding datasets
As an alternative, you can re-build a new dataset using our methodology by executing the following command:
```bash
# Build our NQopen dataset
python make_datasets.py --dataset_name=NQopen --n_distractor_docs=10000  --output_name=NQopen
```

#### Optimize Documents
To generate optimized documents, you can use the following command:
```bash
python make_optimized_documents.py --dataset_path=<path to the dataset folder> --optimization_type=<optimization type>
```
This will spawn algorithm-specific docker containers to run the optimizations.
If you want to run these optimizations on a SLURM cluster, you have to re-build the docker images manually. 

### Benchmarking: 
1) Create a config file to describe the benchmark you want to run and save it in ```Config/Pipelines```. This benchmark will specify:
    - The dataset to use
    - The pipeline to test
See ```Configs/Pipelines/benchmark.yaml``` for an example

2) Run the benchmark using the following command:
```bash
python benchmark_pipeline.py --config-name <name of your configuration file>
```

### Regenerate plots on the paper:
Follow the instructions in `artifact_instuctions.md`
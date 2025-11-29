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

### Dataset
This step is only necessary if you want to re-generate the benchmarks used in this paper or if you want to import your own dataset.
If you want to re-use the same benchmarks used in the paper, you can skip this step and pass to the benchmark section. 

#### Rebuilding datasets
You can re-build our own dataset by executing the following command:
```bash
# Build our NQopen dataset
python make_datasets.py --dataset_name=NQopen --n_distractor_docs=10000  --output_name=NQopen
```

### Benchmark
You can find the benchmarks used in the paper [here](https://mega.nz/file/5P0liAwY#btMh1OOoKJaQbwzShbr79s0sxh5wv_Qj7cTyxMF1pTo). 
Download and unpack them inside the folder <project_root>/data/bechmarks.

Alternatively, to generate a new benchmark, you can re-use the same procedure as in the paper by executing the following command:
```bash
python make_benchmark.py --dataset_path=<path to the dataset folder> --output_name=<output name>  --n_mutations=5 --n_adversarial_docs=5 --n_benign_docs=5
```


#### Optimized Documents
To generate optimized documents, you can use the following command:
```bash
python make_optimized_documents.py --dataset_path=<path to the dataset folder> --optimization_type=<optimization type>
```
This will spawn algorithm-specific docker containers to run the optimizations.

### Benchmarking: 
1) Create a config file to describe the benchmark you want to run and save it in ```Config/Pipelines```. This benchmark will specify:
    - The dataset to use
    - The pipeline to test
See ```Configs/Pipelines/benchmark.yaml``` for an example

2) Run the benchmark using the following command:
```bash
python benchmark.py --config-name <name of your configuration fil (or empty to use the default one)>
```


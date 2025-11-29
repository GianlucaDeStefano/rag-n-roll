# Artifact
These instructions help to reproduce the results seen on the paper. 

### Download the data 
Download the results from (here)[https://mega.nz/file/RW8GhC7C#qPBmmdj0m0fb4iKQr7CTrZEltNf_K0Q-Wg9ezFBMtmQ] and unpack the zip inside the folder <project_root>/data

### Dataset
Table 4 of the paper shows the dataset statistics. 
To reproduce them: 

### Evaluation
Here are the instructions to reproduce the data shown in the evaluation section of the paper.

#### Table 5
Go in analysis/analysis_baselines.ipynb and run the notebook.

#### Table 6
Go in analysis/analysis_attacks_statistics.ipynb and run the notebook.
The first table in the section 'ASR statistics' will give you the results for Table 6 of the paper.

#### Table 8 (correlation coefficients)
Go in analysis_parameters.ipynb and run the notebook.
The first table in the section 'Correlation coefficients' will give you the results for Table 8 of the paper.

#### Table 9 (optimal conf)
Go in analysis_optimal_conf.ipynb and run the notebook.

#### Figure 5 (several malicious documents)
```
python plot_results.py --benchmark_root=./data/results/nqopen_small/benchmark_num_docs  --plot_type=numdocs
```

#### Figure 6 (several benign vs several malicious documents)
```
	python plot_results.py --benchmark_root=./data/results/nqopen_small/benchmark_num_ben  --plot_type=numben
```
#### Figure 7 (question bias)
```
    python plot_results.py --benchmark_root=./data/results/nqopen_small/question_bias --plot_type=bias
```

#### Figure 7 and 8 (parameters graphs)
From the root of the project, run the following command to generate the plots for Figure 7 and Figure 8 of the paper:
```
python plot_results.py --benchmark_root=./data/results/nqopen_small/parameters  --plot_type=parameters
```

#### Figure 9 (LLM generations)
```
python plot_results.py --benchmark_root=./data/results/nqopen_small/model_generations  --plot_type=model_generations
```


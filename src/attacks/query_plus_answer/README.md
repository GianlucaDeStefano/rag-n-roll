# Query+ 
The query plus answer attack is a simple attack where adversarial documents are created by simply adding query and answer to this template:
```
The answer to: '<question>' is '<answer>'.
```
For each query it is possible to create as many adversarial documents as we have adversarial answers.

Use the following command to generate the optimized documents:
```bash
python optimize_dataset.py --dataset_path=<path to the dataset folder> --opt_name=query+answer --placement_strategy=<prefix/proximity>
```
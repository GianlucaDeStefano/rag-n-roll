# Query+ 
The query+ attack optimizes every document by simply adding the query to it. 

We support 2 different placement strategies for the query:
- `prefix`: The query is added at the beginning of the document.
- `proximity`: the query is added in the proximity of the adversarial answers.  

Use the following command to generate the optimized documents:
```bash
python optimize_dataset.py --dataset_path=<path to the dataset folder> --opt_name=query+ --placement_strategy=<prefix/proximity>
```
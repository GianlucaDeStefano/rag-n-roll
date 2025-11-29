## How to Create Your Own Dataset
To use your own dataset with this tool, you need to prepare your data in a specific format that can be easily loaded using the Dataset.load() function.

Below are the steps and guidelines to help you prepare your dataset.

### 1. Dataset Structure
Your dataset will consist of two main components:
1) Documents: Represented by the CorpusObj class.
2) Queries: Represented by the QueryObj class.

To prepare the dataset for loading, you should create two csv files containing the documents and queries, respectively and position them in a folder named as the dataset. 

#### Document Structure (CorpusObj)
Each document in your dataset should be represented as an entry in `corpus.csv` object with the following fields:

- id: A unique identifier for the document (string, integer, or any other hashable type).
title: The title of the document (string).
- text: The full text content of the document (string).
is_adversarial: A boolean value indicating whether the document is adversarial.
- type: A string denoting the origin of the document (e.g., "dataset", "crawled").
- optimization_type (optional): A string indicating the technique used to create an optimized version of the document (e.g., "attack-name").
- related_queries: A list of query IDs that the document is related to. This list must be empty if the document is irrelevant, or contain a single query ID if the document is relevant.
- original_bening_corpus_id (optional): If the document is generated from another document, this field contains the ID of the original document.
- counter: An integer representing the instance of the document (e.g., the first, second, third) spawned from the same parent document.
- metadata: A dictionary containing any additional metadata associated with the document.

Example document representation in JSON format:
```
{
  "id": 1,
  "title": "Document Title",
  "text": "This is the content of the document.",
  "is_adversarial": false,
  "type": "dataset",
  "optimization_type": null,
  "related_queries": [101],
  "original_bening_corpus_id": null,
  "counter": 0,
  "metadata": {}
}
```

#### Query Structure (QueryObj)
Each query in your dataset should be represented as an entry in he `queries.csv` object with the following fields:

- id: A unique identifier for the query (string, integer, or any other hashable type).
- text: The text of the question or query to be asked (string).
- answers: A list of ground truth answers for the query (list of strings).
- text_mutated: A mutated version of the question text (string).
- answers_mutated: A list of mutated ground truth answers (list of strings).
- adversarial_answers: A list of adversarial answers to the query (list of strings).
- adversarial_answers_mutated: A list of mutated adversarial answers (list of strings).
- related_corpuses: A list of document IDs related to the query (list of strings or integers).
- metadata: A dictionary containing any additional metadata associated with the query.
Example query representation in JSON format:

```json
{
  "id": 101,
  "text": "What is the capital of France?",
  "answers": ["Paris"],
  "text_mutated": "What is the capital of France?",
  "answers_mutated": ["Paris"],
  "adversarial_answers": ["Lyon"],
  "adversarial_answers_mutated": ["Lyon"],
  "related_corpuses": [1],
  "metadata": {}
}
```

2. Loading Your Dataset
Once your dataset is properly formatted and saved, you can verify its proper structure by loading it using the Dataset.load() function:

```python

from src.Datasets.BaseDataset import Dataset

dataset = Dataset.load("path/to/dataset/folder")
```

If the dataset loads correctly, you can use it for the benchmark.
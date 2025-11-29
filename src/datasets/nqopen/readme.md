This dataset is based on standard datasets to evaluate open-QA systems called NQ-open (subset of the NQ dataset).
It is built following the methodology described in https://arxiv.org/abs/2307.03172

- The queries are form the NQ-open dataset, filtered to only keep short answers made only of text (not tables) of 5 tokens or less. 
In practice we reused the queries made available by https://arxiv.org/abs/2307.03172 
and downloaded from: https://github.com/nelson-liu/lost-in-the-middle/tree/main/qa_data#:~:text=2%20weeks%20ago-nq%2Dopen%2Doracle.jsonl.gz pairing them with corpuses from the original NQ dataset.



### Download
Download the necessary files:
```
    ./download.sh
```

Build the dataset 

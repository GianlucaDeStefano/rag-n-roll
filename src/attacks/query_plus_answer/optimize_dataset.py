"""
    This attack implements the Query + attack that takes the query the attacker is targeting and places
    in the document it wants to optimize
"""
import argparse
import hashlib
import json
import re
from venv import logger
import os
import copy
from tqdm import tqdm

optimization_name = "query+answer"

### UTILITIES START ###
def load_dataset(dataset_root): 
    """
    This function loads a dataset from the given path.
    
    Args:
        dataset_root (str): path to the root folder of the dataset
    """
    
    # Dataset structure
    dataset = {
        'queries': {},
        'corpuses': {}
    }
    
    # Load queries
    queries_file = f'{dataset_root}/queries.jsonl'
    for line in open(queries_file):
        query = json.loads(line)
        dataset['queries'][query['id']] = query
    
    # Load corpuses
    corpuses_file = f'{dataset_root}/corpus.jsonl'
    for line in open(corpuses_file):
        corpus = json.loads(line)
        dataset['corpuses'][corpus['id']] = corpus
    
    return dataset
### UTILITIES END ###

DOC_TEMPLATE = "The answer to: '{question}' is '{answer}'."

def optimize_context(question, answer, document):
    

    document_text = DOC_TEMPLATE.format(question=question, answer=answer)
    
    # Update the fields in the document object
    document['id'] = str(hashlib.sha256(document_text.encode()).hexdigest())
    document['text'] = document_text
    document['parent_corpus_id'] = None
    document['opt_metadata'] = {}
    document['optimization_type'] = optimization_name
    
    return document


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,required=True,help='Dataset of contexts to optimize')
    parser.add_argument('--n_documents', type=int, default=1, help='Number of documents per query to optimize (selected using the counter field)')
    
    args = parser.parse_args()
    
    # Name of the optimization strategy used
    optimization_name = f'{optimization_name}'
    
    assert os.path.exists(args.dataset_path), "Dataset path does not exist"
    assert os.path.isdir(args.dataset_path), "Dataset path must be a folder"
    
    output_path = f'{args.dataset_path}/additional_corpuses/{optimization_name}.json'
    assert not os.path.exists(output_path), f"Output path {output_path} already exists"
    
    logger.info("Optimizing dataset at {}".format(args.dataset_path))
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Optimize dataset
    queries = {}
    opt_docs = {}
    
    logger.info("Optimizing contexts...")
    
    # Iterate over the queries in the dataset 
    for qid, query in tqdm(dataset['queries'].items()):
        
        possible_answers = query['adversarial_answers']+query['adversarial_answers_mutated']
        
        for c in range(min(args.n_documents,len(possible_answers))):
            
            # Find the document to optimize
            document = None
            for did in query['related_corpuses']:
                if dataset['corpuses'][did]['counter'] == 0 and dataset['corpuses'][did]['is_adversarial'] == True:
                    document = dataset['corpuses'][did]
                    break
            assert document is not None, f"Document with counter {0} not found for query {qid}"
        
            # Optimize the document
            opt_document = optimize_context(query['text'],possible_answers[c], copy.deepcopy(document))

            # Make sure that there is no duplicate document in the dataset
            assert opt_document['id'] not in dataset['corpuses'], f"Duplicate document found: {opt_document['id']}"
            
            # Add the optimized document to the list of optimized docs
            opt_docs[opt_document['id']] = opt_document
            
    # Save the optimized documents to the output path
    os.makedirs(f'{args.dataset_path}/additional_corpuses', exist_ok=True)
    with open(output_path, 'w') as f:
        for doc in opt_docs.values():
            f.write(json.dumps(doc)+'\n')
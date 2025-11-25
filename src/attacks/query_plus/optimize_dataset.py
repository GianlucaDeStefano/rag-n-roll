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
from tqdm import tqdm

optimization_name = "query+"

### UTILITIES START ###

def identify_injection_points(document:str, answers: list, strategy= 'prefix', separator_characters= ['\n','.','!','?']):
    """
        This function identifies the indexes of the injection points in the document where the triggers will be added.
        If the strategy is 'prefix', the triggers are added at the beginning of the document.
        If the strategy is 'proximity', the triggers are added close to the answers in the document. In particular, the triggers are added after the first punctuation mark before the answer.
        
    """
    injection_points = []
      
    assert strategy.lower() in  ['prefix', 'proximity'], "Injection strategy not valid"
    
    if strategy.lower() == 'prefix':
        return [{'separator':'', 'position':0}]
    
    elif strategy.lower() == 'proximity': 
        # find all punctuation marks
        split_position_candidates = []
        
        # Find possible split positions
        for separator in separator_characters:
            for separator_position in re.finditer(re.escape(separator.lower()), document.lower()):
                split_position_candidates.append((separator, separator_position.end()))

        # Sort split positions by position in descending order
        split_position_candidates = sorted(split_position_candidates,key=lambda x: x[1], reverse = True)
        found_answers = 0 
        
        selected_injection_position = []

        # Iterate over all candidate answers
        for answer in answers: 
            
            # find positions of the answer in the document
            answer_positions = [m for m in re.finditer(answer.lower(), document.lower())]

            found_answers += len(answer_positions)
            
            for answer_pos in answer_positions:
                # find the closest separator mark before the answer
                found_injection_pos = False
                for separator,position in split_position_candidates:
                    if position <= answer_pos.start():
                        
                        # If this position has not been selected before, add it to the list of injection points
                        if position not in selected_injection_position:
                            injection_points.append({'separator':separator, 'position':position})
                            selected_injection_position.append(position)

                        found_injection_pos = True
                        break
                
                # If no injection point was found, add the beginning of the document as an injection point
                if not found_injection_pos:
                    if 0 not in selected_injection_position:
                        selected_injection_position.append(0)
                        injection_points.append({'separator':'', 'position':0})
        
        
        # If no answer was found, add the beginning of the document as an injection point
        if found_answers == 0:
            # This can happen for example in datasets like those from PoisonRAG that have pre-generated adversarial documents that 
            # may not contain the exact answer.
            # In this case, we add the beginning of the document as an injection point
            if 0 not in selected_injection_position:
                selected_injection_position.append(0)
                injection_points.append({'separator':'', 'position':0})
                
        return list(injection_points)

    raise ValueError('Invalid injection strategy')

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



def inject_triggers(document_text:str, injection_points: list, trigger: str):
    
    # Sort injection points by position
    injection_points = sorted(injection_points,key=lambda x: x['position'],reverse=True) 
    
    opt_metadata = []

    while len(injection_points) > 0:
        
        # get current injection point
        injection_point = injection_points.pop(0)
        
        # inject triggers
        document_text = document_text[:injection_point['position']] + trigger + ' ' + document_text[injection_point['position']:]
        
        # Update old opt_metadata
        for i in range(len(opt_metadata)):
            opt_metadata[i]['position'] += len(trigger) + 1
        
        opt_metadata.append({
            'position': injection_point['position'],
            'separator': injection_point['separator'],
            'triggers': trigger,
        })
        
    return document_text, opt_metadata

def optimize_context(query, document, injection_strategy):
    
    # List of answers to look for
    answers = query['adversarial_answers'] + query['adversarial_answers_mutated']
    
    # Find injection points in the document
    injection_points = identify_injection_points(document['text'], answers,injection_strategy)       
    
    # Create opt_text
    document_text, opt_metadata = inject_triggers(document['text'], injection_points, query['text'])
    
    # Update the fields in the document object
    document['id'] = str(hashlib.sha256(document_text.encode()).hexdigest())
    document['text'] = document_text
    document['opt_metadata'] = opt_metadata
    document['optimization_type'] = optimization_name
    
    return document


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,required=True,help='Dataset of contexts to optimize')
    parser.add_argument('--injection_strategy', type=str,required=True,help='Where should the triggers be placed? Options are: prefix, proximity')
    parser.add_argument('--n_documents', type=int, default=1, help='Number of documents per query to optimize (selected using the counter field)')
    
    args = parser.parse_args()
    
    # Name of the optimization strategy used
    optimization_name = f'{optimization_name}_{args.injection_strategy}'
    
    assert os.path.exists(args.dataset_path), "Dataset path does not exist"
    assert os.path.isdir(args.dataset_path), "Dataset path must be a folder"
    assert args.injection_strategy in ['prefix', 'proximity'], f"Injection strategy:{args.injection_strategy} not valid, choose between: ['prefix', 'proximity']"
    
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
        
        # Iterate over the counters of the document to optimize (remember that the counters are 0-based)
        for counter in range(args.n_documents):

            # Find the document to optimize
            document = None
            for did in query['related_corpuses']:
                if dataset['corpuses'][did]['counter'] == counter and dataset['corpuses'][did]['is_adversarial'] == True:
                    document = dataset['corpuses'][did]
                    break
            assert document is not None, f"Document with counter {counter} not found for query {qid}"
            
            # Optimize the document
            opt_document = optimize_context(query,document,args.injection_strategy)

            # Make sure that there is no duplicate document in the dataset
            assert opt_document['id'] not in dataset['corpuses'], f"Duplicate document found: {opt_document['id']}"
            
            # Add the optimized document to the list of optimized docs
            opt_docs[opt_document['id']] = opt_document
            
    # Save the optimized documents to the output path
    os.makedirs(f'{args.dataset_path}/additional_corpuses', exist_ok=True)
    with open(output_path, 'w') as f:
        for doc in opt_docs.values():
            f.write(json.dumps(doc)+'\n')
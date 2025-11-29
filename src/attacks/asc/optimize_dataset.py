"""
    This script optimizes the context of a dataset of samples.
    The dataset is a list of samples, where each sample is a dictionary with the following keys:
        - query: The query for which the context is optimized
        - context: The context to optimize
        - optimized_context: The optimized context (to be filled by this script)
    The script will then overwrite the dataset with the new optimized contexts.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import re
from tqdm import tqdm
import logging
from collision_retrieval import gen_aggressive_collision, gen_natural_collision
from transformers import AutoModel, AutoTokenizer
import torch
from models.bert_models import BertForConcatNextSentencePrediction, BertForLM
from models.scorer import SentenceScorer
from transformers import BertTokenizer
logger = logging.getLogger(__name__)
from collision_retrieval import parser
from text_chunker import TextChunker
from rank_bm25 import BM25Okapi

import os 

device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')

### UTILITIES START ###

def find_highest_ranking_chunks_bm25(queries,documents,chunk_size=256,top_n = 100):
    """
        This function finds the highest ranking chunks of the documents for the given queries.
    """ 
    
    chunks = []
    chunker = TextChunker(maxlen=chunk_size)
    for doc in documents:
        chunks += [c.split(' ') for c  in chunker.chunk(doc['text'])]

    bm25 = BM25Okapi(chunks)
    
    top_ranking_chunks = defaultdict(list)
    
    for query in tqdm(queries,desc='Finding highest ranking chunks with BM25'):
        best_passages = bm25.get_top_n(query['text'].split(" "), chunks, n=top_n)
        top_ranking_chunks[query['id']] = [" ".join(p) for p in best_passages]
        
    return top_ranking_chunks

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


def optimize_context(query, doc_text,ben_doc_text, injection_strategy,args):
    
    answers = query['adversarial_answers'] + query['adversarial_answers_mutated']
    injection_points = identify_injection_points(doc_text, answers, injection_strategy)
    injection_points = sorted(injection_points,key=lambda x: x['position'],reverse=True) 
    
    opt_metadata = []
    
    # The input_b in the original paper is a string formed by the highest scoring sentences for the query
    # We instead use the benign_doc id as we assume that the attacker does not have full access to the datasource
    if args.mode == 'natural':
        
        # Default parameters from the repo:
        #(N) python3 collision_retrieval.py --num_beams=10 --topk=150 --perturb_iter=5 --stemp=0.1 --lr=1e-3 --seq_len=35 --num_filters=1000 --nature --beta=0.02 --fp16
        args.num_beams=10
        args.topk=150
        args.perturb_iter=5
        args.stemp=0.1
        args.lr=1e-3 
        args.seq_len=35
        args.num_filters=1000
        args.beta=0.02
        
        collision, new_score, collision_cands = gen_natural_collision(query['text'], '', model, tokenizer, device, lm_model, 0, eval_lm_model,args)
    elif args.mode == 'aggressive':
        
        if args.regularize:
            
            # Default parameters from the repo:
            # (R) python3 collision_retrieval.py --num_beams=5 --topk=40 --perturb_iter=30 --stemp=1.0 --lr=1e-3 --seq_len=60 --num_filters=1000 --regularize --beta=0.85 --fp16
            args.num_beams=5
            args.topk=40 
            args.perturb_iter=30 
            args.stemp=1.0 
            args.lr=1e-3 
            args.seq_len=60 
            args.num_filters=1000 
            args.regularize = True 
            args.beta=0.85 
        
        else:
            
            # Default parameters from the repo:
            # (A) python3 collision_retrieval.py --num_beams=5 --topk=50 --perturb_iter=30 --stemp=1.0 --lr=1e-3 --seq_len=30 --num_filters=1000 --fp16
            args.num_beams=5
            args.topk=50
            args.perturb_iter=30
            args.stemp=1.0
            args.lr=1e-3
            args.seq_len=30
            args.num_filters=1000
            
        collision, new_score, collision_cands = gen_aggressive_collision(query['text'], '', model, tokenizer, device, 0, lm_model,args)
    else:
        raise ValueError(f"Mode {args.mode} not supported")

    if not collision:
        logger.info(f"Failed to generate collision for query {query['text']}")
        return doc_text, opt_metadata
    
    while len(injection_points) > 0:
        
        # get current injection point
        injection_point = injection_points.pop(0)
        
        # inject triggers
        doc_text = doc_text[:injection_point['position']] + collision +' '+ doc_text[injection_point['position']:]
    
        # Update old opt_metadata
        for i in range(len(opt_metadata)):
            opt_metadata[i]['position'] += len(collision) + 1
        
        opt_metadata.append({
            'position': injection_point['position'],
            'separator': injection_point['separator'],
            'triggers': collision,
        })
    
    return doc_text, opt_metadata

if __name__ == "__main__":
    
    parser.add_argument('--dataset_path', type=str,required=True,help='Dataset of contexts to optimize')
    parser.add_argument('--injection_strategy', type=str,required=True,help='Where should the triggers be placed? Options are: Start, Smart')
    parser.add_argument('--mode', type=str,required=True,help='What mode of the attack to run')
    parser.add_argument('--n_documents', type=int, default=1, help='Number of documents per query to optimize (selected using the counter field)')

    args = parser.parse_args()
        
    # If the necessary models are not downloaded, download them
    if not os.path.exists('collision/birch'):
        os.system('chmod +x ./install.sh && ./install.sh')
    
    assert args.mode in ['natural', 'aggressive'], f"Mode {args.mode} not supported, please use 'natural' or 'aggressive'"

    regularization_flag = 'reg' if args.regularize else 'noreg'
    
    optimization_name = f"asc-{args.mode}-{regularization_flag}_{args.injection_strategy}"
        
    assert os.path.exists(args.dataset_path), "Dataset path does not exist"
    assert os.path.isdir(args.dataset_path), "Dataset path must be a folder"
    assert args.injection_strategy in ['prefix', 'proximity'], f"Injection strategy:{args.injection_strategy} not valid, choose between: ['prefix', 'proximity']"
    
    output_path = f'{args.dataset_path}/additional_corpuses/{optimization_name}.json'
    assert not os.path.exists(output_path), f"Output path {output_path} already exists"
    
    logger.info("Optimizing dataset at {}".format(args.dataset_path))
    
    dataset = load_dataset(args.dataset_path)
    
    # Load models
    model_path = os.path.join(args.model_dir, args.model_name)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForConcatNextSentencePrediction.from_pretrained(model_path)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    lm_model = BertForLM.from_pretrained(args.lm_model_dir)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False
    eval_lm_model = SentenceScorer(device)
    
    # Optimize dataset
    opt_docs = {}
    
    logger.info("Optimizing contexts...") 
    
    benign_documents = [doc for doc in dataset['corpuses'].values() if doc['is_adversarial'] == False and doc['counter'] < args.n_documents]
    best_chunks_bm25 = find_highest_ranking_chunks_bm25(dataset['queries'].values(), benign_documents,top_n=1)
    
    i = 0 
    for qid, query in tqdm(dataset['queries'].items()):
        
        for counter in range(args.n_documents):
            
            adv_document = None
            for did in query['related_corpuses']:
                if dataset['corpuses'][did]['counter'] == counter and dataset['corpuses'][did]['is_adversarial'] == True:
                    adv_document = dataset['corpuses'][did]
                    break
            assert adv_document is not None, f"Document with counter {counter} not found for query {qid}"
            
            # Input_b is a string formed by the highest ranking sentences in the original dataset of the paper
            # here we use the highest ranking chunks belonging to the benign portion of the dataset
            optimized_context, opt_metadata = optimize_context(query, adv_document['text'],best_chunks_bm25[qid][0], args.injection_strategy,args)

            new_id = str(hashlib.sha256(optimized_context.encode()).hexdigest())
            adv_document['text'] = optimized_context
            adv_document['optimization_type'] = optimization_name
            adv_document['opt_metadata'] = opt_metadata
            adv_document['parent_corpus_id'] = adv_document['id']
            adv_document['id'] = new_id
            opt_docs[adv_document['id']] = adv_document
        
    # Save the optimized documents to the output path
    os.makedirs(f'{args.dataset_path}/additional_corpuses', exist_ok=True)
    with open(output_path, 'w') as f:
        for doc in opt_docs.values():
            f.write(json.dumps(doc)+'\n')
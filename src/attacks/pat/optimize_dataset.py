
from argparse import Namespace
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import re
from venv import logger
from adv_ir.attack_methods import pairwise_anchor_trigger
from bert_ranker.models import pairwise_bert
import torch
import os
from tqdm import tqdm
import nltk
from bert_ranker.models.bert_lm import BertForLM
from apex import amp
from transformers import BertTokenizerFast, BertForNextSentencePrediction
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import nltk
from rank_bm25 import BM25Okapi
from text_chunker import TextChunker
nltk.download('stopwords')

device = torch.device("cuda")
optimization_name = "PAT"

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

def load_surrogate_model(transformer_model, model_path = 'data/Imitation.MiniLM.L12.v2.BertForPairwiseLearning.bert-base-uncased.pth'):
    model = pairwise_bert.BertForPairwiseLearning.from_pretrained(transformer_model)
    t = torch.load(model_path,map_location='cpu')
    model.load_state_dict(t)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)

def load_lm_model(lm_model_dir):
    lm_model = BertForLM.from_pretrained(lm_model_dir)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False
    return lm_model

def load_nsp_model(transformer_model):
    nsp_model = BertForNextSentencePrediction.from_pretrained(transformer_model)
    nsp_model.to(device)
    nsp_model.eval()
    for param in nsp_model.parameters():
        param.requires_grad = False
    return nsp_model        

# Tokenizer and model for selecting the anchors
anchor_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
anchor_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2").to(device)

def find_anchors(queries,documents): 
    
    def rerank_chunks(query_text,chunks, batch_size=32):

        all_logits = []
        l = [(query_text,chunk) for chunk in chunks]
        for index in range(0,len(l),batch_size):
            batch_encoding = anchor_tokenizer(l[index:index+batch_size], padding="max_length", truncation=True,return_tensors='pt')

            for k in batch_encoding.keys():
                batch_encoding[k] = batch_encoding[k].to(device)
            
            outputs = anchor_model(**batch_encoding)

            scores = outputs.logits.squeeze()
            if len(l[index:index+batch_size])  > 1:
                all_logits += scores.tolist()
            else:
                all_logits.append(scores.item())
                
        sorted_docs = [x for _, x in sorted(zip(all_logits, chunks),reverse=True)]
        return sorted_docs

    # Find the 1000 best chunks using BM25
    best_chunks_bm25 = find_highest_ranking_chunks_bm25(queries,documents,top_n=1000)
    
    # Rerank the chunks using the surroqate model
    anchor_passages = {}
    for query in queries:
        anchor_passages[query['id']] = rerank_chunks(query['text'],best_chunks_bm25[query['id']])
        
    return anchor_passages    


def optimize_context(query, anchors,adv_document, injection_strategy,attack_args):
    
    question = query['text']
    document_2_opt = adv_document['text']
        
    answers = query['adversarial_answers'] + query['adversarial_answers_mutated']
    injection_points = identify_injection_points(document_2_opt, answers,injection_strategy)
    injection_points = sorted(injection_points,key=lambda x: x['position'],reverse=True) 
    
    opt_metadata = []
    
    # As the paper we use the top-3 ranking documents
    anchors = " ".join(anchors[:3])
    
    while len(injection_points) > 0:
        
        # get current injection point
        injection_point = injection_points.pop(0)
        
        # compute triggers
        best_trigger,_ ,_ = pairwise_anchor_trigger(query=question, anchor=anchors, raw_passage=document_2_opt[injection_point['position']:injection_point['position'] + 1024], model=model, tokenizer=tokenizer, device=device, args=attack_args, lm_model=lm_model,nsp_model=nsp_model)
            
        # inject triggers
        document_2_opt = document_2_opt[:injection_point['position']] + best_trigger + ' ' + document_2_opt[injection_point['position']:]

        # Update old opt_metadata
        for i in range(len(opt_metadata)):
            opt_metadata[i]['position'] += len(best_trigger) + 1
        
        opt_metadata.append({
            'position': injection_point['position'],
            'separator': injection_point['separator'],
            'triggers': best_trigger,
        })
        
    return document_2_opt, opt_metadata

if __name__ == "__main__":
    
    #Defaults param from the repo:
    # --target=mini --imitation_model=imitate.v2  --nsp --lambda_1=0.6 --lambda_2=0.1 --num_beams=10 --topk=128 --mode=train
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,required=True,help='Dataset of contexts to optimize')
    parser.add_argument('--injection_strategy', type=str,required=True,help='Where should the triggers be placed? Options are: Start, Smart')
    parser.add_argument('--n_documents', type=int, default=1, help='Number of documents per query to optimize (selected using the counter field)')
    
    parser.add_argument("--target", type=str, default='mini', help='test on what model')
    parser.add_argument("--imitation_model", default='imitate.v2', type=str)
    parser.add_argument("--data_name", default="dl", type=str)
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--tri_len", default=6, type=int, help="Maximun trigger length for generation.")
    parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
    parser.add_argument("--topk", default=128, type=int, help="Top k sampling for beam search")
    parser.add_argument('--max_iter', type=int, default=20, help='maximum iteraiton')
    parser.add_argument("--lambda_1", default=0.6, type=float, help="Coefficient for language model loss.")
    parser.add_argument('--stemp', type=float, default=0.1, help='temperature of softmax')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='penalty of repetition')
    parser.add_argument('--lr', type=float, default=0.001, help='optimization step size')
    parser.add_argument("--num_beams", default=10, type=int, help="Number of beams")
    parser.add_argument("--num_sims", default=300, type=int, help="Number of similar words")
    parser.add_argument('--perturb_iter', type=int, default=5, help='PPLM iteration')
    parser.add_argument("--seed", default=42, type=str, help="random seed")
    parser.add_argument("--nsp", action='store_true', default=True)
    parser.add_argument("--lambda_2", default=0.1, type=float, help="Coefficient for language model loss.")

    # Support setting
    parser.add_argument("--lm_model_dir", default='data/bert', type=str,
                        help="Path to pre-trained language model")
    
    args = parser.parse_args()
    
    optimization_name = f'PAT_{args.injection_strategy}'
    output_path = f'{args.dataset_path}/additional_corpuses/{optimization_name}.json'

    if not os.path.exists(f'./data'):
        os.system('chmod +x ./install.sh && ./install.sh')

    if args.imitation_model == 'imitate.v2':
        model_path = './data/Imitation.MiniLM.L12.v2.BertForPairwiseLearning.bert-base-uncased.pth'
    elif args.imitation_model == 'imitate.v1':
        model_path = './data/Imitation.bert_large.BertForPairwiseLearning.bert-base-uncased.pth'
    else:
        model_path = args.imitation_model

    tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
    model,lm_model, nsp_model = [load_surrogate_model(args.transformer_model,model_path),load_lm_model(args.lm_model_dir),load_nsp_model(args.transformer_model)]

    logger.info("Optimizing dataset at {}".format(args.dataset_path))
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Find best passages for each query
    anchors = find_anchors(dataset['queries'].values(),[doc for doc in dataset['corpuses'].values() if not doc['is_adversarial'] and doc['counter'] < args.n_documents])
    
    # Optimize dataset
    opt_docs = {}
    
    logger.info("Optimizing contexts...") 
    for  qid,query in tqdm(dataset['queries'].items()):
        
        for c in range(args.n_documents):
            
            adv_document = None
            for did in query['related_corpuses']:
                if dataset['corpuses'][did]['counter'] == c and dataset['corpuses'][did]['is_adversarial'] == True:
                    adv_document = dataset['corpuses'][did]
                    break
            
            assert adv_document is not None, f"Adversarial document with counter {c} not found for query {qid}"

            optimized_context, opt_metadata = optimize_context(query,anchors[qid],adv_document,args.injection_strategy,args)
            
            new_adv_doc = copy.deepcopy(adv_document)
            
            new_adv_doc['text'] = optimized_context
            new_adv_doc['opt_metadata'] = opt_metadata
            new_adv_doc['optimization_type'] = optimization_name
            new_adv_doc['id'] = hashlib.sha256(optimized_context.encode()).hexdigest()
            
            opt_docs[new_adv_doc['id']] = new_adv_doc
    
    # Save the optimized documents to the output path
    os.makedirs(f'{args.dataset_path}/additional_corpuses', exist_ok=True)
    with open(output_path, 'w') as f:
        for doc in opt_docs.values():
            f.write(json.dumps(doc)+'\n')
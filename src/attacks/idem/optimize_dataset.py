from collections import defaultdict
import hashlib
import json 
import os 
import re
from tqdm import tqdm 
import argparse
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

files_folder = '/tmp'
chunk_size = 1024 # If the document is larger than this, we chunk it into smaller parts to optimize


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

### UTILITIES END ###

def format_text(text):
    # We replace new lines with <br> to avoid it being escaped by the nlp library, we will replace it back later
    text = text.replace('\n','<br>')
    
    # We do this to preventively format the texts in a way that  avoid the case where the document is too large and the nlp library crashes
    text_sentences = list(nlp(text).sents)
    text = "".join([str(s) for s in text_sentences])
    return text

def chunk_text(text,chunk_size):
    chunk_sents = list(nlp(text).sents)
    chunk = ""
    
    for s in chunk_sents:
        if len(chunk) + len(s) > chunk_size:
            break
        chunk += str(s)

    if not chunk:
        chunk = text[:chunk_size]
    return chunk

def generate_connection_sentences(dataset, data_folder, injection_strategy,n_documents = 1): 
    
    # We have to generate the files to call the script ./generate_connection_sents.py
    # the files are:
    # --target_file : file containing the association: <query_id> <doc_id> <rank> <similarity_score>
    # --query_collection: file containing the association: <query_id>\t<query_text>
    # --doc_collection: file containing the association: <doc_id>\t<doc_text>
        
    os.makedirs(data_folder,exist_ok=True)
    
    query_ids = []
    query_collection_file = f'{data_folder}/query_collection.txt'
    query_collection = []
    for query_id, query in dataset['queries'].items():
        
        query_collection.append({'id':query_id,'text':query['text']})
        query_ids.append(query_id)
    pd.DataFrame(query_collection).to_csv(query_collection_file,sep='\t',index=False,header=False)
      
    doc_2_query = {}
    for query_id, query in dataset['queries'].items():
        
        for doc_id in list(set(query['related_corpuses'])):
            assert doc_id not in doc_2_query, f"Document {doc_id} is related to more than one query"
            doc_2_query[doc_id] = query
        
    doc_collection_file = f'{files_folder}/doc_collection.txt'
    doc_collection = []
    query_docs = defaultdict(lambda: [])       
    for doc_id, doc in dataset['corpuses'].items():
        
        # Skip non-adversarial documents or documents that have ha counter >= n_documents
        if not doc['is_adversarial'] or doc['counter'] >= n_documents:
            continue
        
        assert doc_id in doc_2_query, f"Document {doc_id} is not related to any query: {doc}"
        text = format_text(doc['text'])
        
        # Create list of adversarial answers
        answers = doc_2_query[doc_id]['adversarial_answers']+ doc_2_query[doc_id]['adversarial_answers_mutated']
        
        # Identify injection points
        injection_points = identify_injection_points(text,answers,injection_strategy)
        
        print(injection_points)
        
        for injection_point in injection_points:
            
            chunk = chunk_text(text[injection_point['position']:],chunk_size)
            
            doc_collection.append({'id':f"{doc_id}_{injection_point['position']}",'text':chunk})
            query_docs[doc_2_query[doc_id]['id']] += [f'{doc_id}_{injection_point["position"]}']
            
    pd.DataFrame(doc_collection).to_csv(doc_collection_file,sep='\t',index=False,header=False)
    
    
    target_file = f'{files_folder}/target_file.txt'
    targets = []
    with open(target_file, 'w') as f:
        for query_id, doc_ids in query_docs.items():
            if query_id not in query_ids:
                continue
            for doc_id in doc_ids:
                targets.append({'query_id':query_id,'doc_id':doc_id,'rank':0,'similarity_score':0})
    pd.DataFrame(targets).to_csv(target_file,sep='\t',index=False,header=False)
    
    connection_sentences_file = f'{files_folder}/connection_sentences.txt'
    # Now we call the script
    os.system(f'CUDA_VISIBLE_DEVICES=0 python3 generate_connection_sents.py --target_file {target_file} --query_collection {query_collection_file} --doc_collection {doc_collection_file} --output_connect_sents {connection_sentences_file}')

    return query_collection_file, doc_collection_file , target_file, connection_sentences_file

def generate_adv_documents(surrogate_model, surrogate_tag, connect_sent_file, target_file, query_collection, doc_collection):
    
    os.system(f"""CUDA_VISIBLE_DEVICES=0 python -u merge_connection_sents_with_docs.py \
                              --surrogate_model {surrogate_model} \
                              --connect_sent_file {connect_sent_file} \
                              --target_file {target_file} \
                              --query_collection {query_collection} \
                              --doc_collection {doc_collection} \
                              --coh_weight 0.5 \
                              --rel_weight 0.5 \
                              --batch_size 100 """)


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,required=True,help='Dataset of contexts to optimize')
    parser.add_argument('--injection_strategy', type=str,required=True,help='Where should the triggers be placed? Options are: Start, Smart')
    parser.add_argument('--n_documents', type=int, default=1, help='Number of documents per query to optimize (selected using the counter field)')
    parser.add_argument('--surrogate_model', type=str,default='S1',help='Name of the surrogate model to use')
    
    args = parser.parse_args()
    optimization_name = f'IDEM_{args.injection_strategy}'
    
    if not os.path.exists(f'data/surrogate_models/'):
        os.system('chmod +x ./install.sh && ./install.sh')
    
    assert os.path.exists(args.dataset_path), "Dataset path does not exist"
    assert os.path.isdir(args.dataset_path), "Dataset path must be a folder"
    assert args.injection_strategy in ['prefix', 'proximity'], f"Injection strategy:{args.injection_strategy} not valid, choose between: ['prefix', 'proximity']"
    
    output_path = f'{args.dataset_path}/additional_corpuses/{optimization_name}.json'
    assert not os.path.exists(output_path), f"Output path {output_path} already exists"

    dataset = load_dataset(args.dataset_path)
    
    # Generate connection sentences
    query_collection_file, doc_collection_file , target_file, connection_sentences_file = generate_connection_sentences(dataset,files_folder, args.injection_strategy,args.n_documents)
    
    # Generate the final adversarial documents
    generate_adv_documents(f'data/surrogate_models/{args.surrogate_model}', '', connection_sentences_file, target_file, query_collection_file, doc_collection_file)
    
    # Generate final adversarial documents
    connection_sentences = pd.read_csv(f'{files_folder}/connection_sentences.txt.position.coh0.5-rel0.5-top1.qry-sent.tsv',sep='\t',names=["query_id", "DocId", "Query", "Sentence"])
    connection_sentences.loc[connection_sentences['Sentence'].isna(), 'Sentence'] = ''
    def extract_ids(record): 
        
        doc_id, metadata = record['DocId'].split('_')
        
        if not record['Sentence']:
            # No valid sentence was generated
            return {'doc_id':doc_id,'char_shift':-1,'sentence_id':-1}
        
        char_shift,sentence_id, injection_sentence_shift = metadata.split('-')
        
        assert int(char_shift) >= 0, f"Error, char_shift is negative: {char_shift} but it should be >= 0 for doc_id: {doc_id}"
        
        adversarial_document = dataset['corpuses'][doc_id]
        formatted_text = format_text(adversarial_document['text'])
        opt_document_span = chunk_text(formatted_text[int(char_shift):],chunk_size)

        sentences = list(nlp(opt_document_span).sents)
        total_shift = int(char_shift)

        for i in range(int(injection_sentence_shift)):
            total_shift += len(str(sentences[i]))
                
        return {'doc_id':doc_id,'char_shift':total_shift,'sentence_id':sentence_id}


    df = connection_sentences.apply(lambda x: extract_ids(x), axis='columns', result_type='expand')
    connection_sentences = pd.concat([df, connection_sentences], axis='columns').sort_values(by=['doc_id', 'char_shift'], ascending=[True, False])
    connection_sentences = connection_sentences.groupby(['doc_id','query_id']).agg({'char_shift': list, 'Sentence': list}).reset_index()

    opt_docs = {}
    
    queries_with_adv_doc = []
    
    missing_query_ids = [q['id'] for q in dataset['queries'].values()]
    
    for _,record in connection_sentences.iterrows():
        
        document = dataset['corpuses'][record['doc_id']]
        query = dataset['queries'][record['query_id']]
        
        missing_query_ids.remove(record['query_id'])
        
        queries_with_adv_doc.append(record['query_id'])
        
        opt_metadata = []
        newly_opt_doc = format_text(document['text'])
                
        sentences = record['Sentence']
        char_shifts = [int(c) for c in record['char_shift']]
        
        assert len(sentences) == len(char_shifts), f"Error, different number of sentences and char_shifts: {len(sentences)} {len(char_shifts)}"
        
        # Order in descending order
        char_shifts,sentences = zip(*sorted(zip(char_shifts, sentences),reverse=True))
        
        print(newly_opt_doc)
        
        print('Initial shifts: ', char_shifts)
        
        used_char_shifts = []     
        for i in range(len(sentences)):

            if char_shifts[i] == -1 or char_shifts[i] in used_char_shifts:
                continue
            
            if i > 0:
                assert char_shifts[i] <= char_shifts[i-1], f"Error, shifts are not ordered! {record['char_shift'][i]} {record['char_shift'][-1]}"
                if  char_shifts[i] == char_shifts[i-1]:
                    continue
            
            print('At shift:', char_shifts[i], 'Sentence:', sentences[i])
             
            newly_opt_doc = newly_opt_doc[:char_shifts[i]] + sentences[i] +' '+ newly_opt_doc[char_shifts[i]:]
                        
            # Update old positions
            for c in range(len(opt_metadata)):
                opt_metadata[c]['position'] = int(opt_metadata[c]['position']) + len(sentences[i]) + 1
                        
            # Add new metadata
            opt_metadata.append({
                'position': int(char_shifts[i]),
                'triggers': str(sentences[i]),
            })
            
        document['text'] = newly_opt_doc.replace('<br>','\n')
        document['opt_metadata'] = opt_metadata
        document['optimization_type'] = optimization_name
        document['id'] = hashlib.sha256(newly_opt_doc.encode()).hexdigest()
                
        opt_docs[document['id']] = document
    
    os.makedirs(f'{args.dataset_path}/additional_corpuses', exist_ok=True)
    with open(output_path, 'w') as f:
        for doc in opt_docs.values():
            f.write(json.dumps(doc)+'\n')
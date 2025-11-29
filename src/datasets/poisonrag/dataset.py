from collections import defaultdict
import os 
from beir import util
import pandas as pd
import json 
from src.pipelines.evaluate.utilities import normalize_str, verify_presence_multi,verify_presence
from src.datasets.base_dataset import CorpusObj, Dataset, QueryObj
import hashlib
from tqdm import tqdm
DATASETS = ['nq','msmarco','hotpotqa']
THIS_FOLDER_PATH = "/".join(__file__.split('/')[:-1])

ANSWERS_2_DISCARD = ['yes','no','true','false']
MIN_ANSWWER_LEN = 4

def make_poisonrag_dataset(dataset_name, output_folder = 'datasets', **kwargs):
    """
    Create a dataset starting from those from PoisonRAG
    """
    
    dataset = dataset_name.split('-')[1]
    
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)

    beir_dataset_folder = os.path.join(output_folder,dataset)
    if not os.path.exists(beir_dataset_folder):
        util.download_and_unzip(url, output_folder)
        
    # load qrels 
    original_qrels = defaultdict(list) # query_id -> [doc_id1, doc_id2, ...]
    for split in ['train','dev','test']:
        qrels_file = os.path.join(beir_dataset_folder,'qrels', '{}.tsv'.format(split))
        
        if not os.path.exists(qrels_file):
            continue
        
        qrels = pd.read_csv(qrels_file, sep='\t', header=None)
        for i, row in qrels.iterrows():
            
            if row[0] == 'query-id':
                continue
            
            # Check that the score is positive (i.e. the document is relevant to the query explanation https://github.com/beir-cellar/beir/issues/80) 
            if int(row[2]) > 0:
                original_qrels[row[0]].append(row[1])

    # Load the original queries
    queries_file = os.path.join(beir_dataset_folder, 'queries.jsonl')
    question_2_original_id = defaultdict(list) # question -> [qid1, qid2, ...] # we have to support questions with multiple ids due to hotspotqa
    with open(queries_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            question_2_original_id[line['text']].append(line['_id'])
    
    # Load the adversarial queries and format them in our QueryObj format
    adv_queries_path = f'{THIS_FOLDER_PATH}/{dataset}.json'
    adv_queries_data = json.load(open(adv_queries_path))
    queries = {} # qid -> QueryObj
    corpuses = {} # doc_id -> CorpusObj
    doc_rels = defaultdict(list) # doc_id -> [new_qid1, new_qid2, ...]
    
    # For each adversarial query
    for qid, query_data in adv_queries_data.items():
        
        # Check that the question is present in the original queries
        assert query_data['question'] in question_2_original_id
        
        # Construct the query object
        new_qid = str(hashlib.sha256(query_data['question'].encode('utf-8')).hexdigest())
        query = QueryObj(id=new_qid, text=query_data['question'], answers=[query_data['correct answer']],adversarial_answers=[query_data['incorrect answer']], metadata={'original_dataset_id':query_data['id']}, related_corpuses=[])
                
        # if the answer is in the list of answers to discard or if it is too short we skip the query
        if query.answers[0].lower() in ANSWERS_2_DISCARD or len(query.answers[0]) < MIN_ANSWWER_LEN:
            continue
        
        # Add the new qid of the query to the list of relevant queries for the original documents
        for original_qid in question_2_original_id[query_data['question']]:
            for doc_id in original_qrels[original_qid]:
                assert new_qid not in doc_rels[doc_id]
                doc_rels[doc_id].append(new_qid)
                assert len(doc_rels[doc_id]) == 1
                
        # Add the adversarial documents associated with this query
        for i, adv_text in enumerate(query_data['adv_texts']):
            # Create the corpus object
            corpus = CorpusObj(id=str(hashlib.sha256(adv_text.encode('utf-8')).hexdigest()) , title="", text=adv_text, is_adversarial=True, type='dataset', related_queries=[new_qid], parent_corpus_id=None,counter=i, metadata={})
            # Make sure that the document is not already present in the dataset
            assert corpus.id not in corpuses
            # Add the document id to the list of related documents for the query
            query.related_corpuses.append(corpus.id)
            # Add the document to the dataset
            corpuses[corpus.id] = corpus
        
        # Make sure that the query is not already present in the dataset
        assert query.id not in queries
        queries[query.id] = query
    
    ##
    ## At this point we have loaded all the adversarial queries and documents
    ## Now we load the original documents
    ##
    corpus_file = os.path.join(beir_dataset_folder, 'corpus.jsonl')
    
    # Load the corpuses
    with open(corpus_file, 'r') as f:
        for line in tqdm(f,desc='Loading corpus documents'):
            
            # Parse the line containing the corpus
            line = json.loads(line)
                        
            # Load metadata and save original dataset id
            metadata = line['metadata']
            metadata['original_dataset_id'] = line['_id']
            
            # Compute the new id hashing the text of the document
            #new_id = str(hashlib.sha256(line['text'].encode('utf-8')).hexdigest())
            new_id = line['_id']
            
            # Create the corpus object
            corpus = CorpusObj(id=new_id,title=line['title'],text=line['text'],is_adversarial=False,type='dataset',metadata=metadata)
            
            # Assert that this document is relevant to at most one query
            assert len(doc_rels[line['_id']]) <= 1, "Document {} is relevant to multiple queries: {}".format(line['_id'],doc_rels[line['_id']])
            
            # Populate the related queries field 
            corpus.related_queries = doc_rels[line['_id']]
                                
            # Add the document to the list of related documents for the queries
            for qid in corpus.related_queries:
                assert new_id not in queries[qid].related_corpuses, "Document {} already present in query {}".format(new_id,qid)
                corpus.counter = len([did for did  in queries[qid].related_corpuses if not corpuses[did].is_adversarial]) 
                queries[qid].related_corpuses.append(new_id)
            
            assert new_id not in corpuses, "Document {} already present in the dataset with id: {}".format(line['_id'],corpuses[new_id].metadata['original_dataset_id'])
            
            # Add the document to the dataset
            corpuses[new_id] = corpus

    # Assert that every document is related to at most one query
    for doc in corpuses.values():
        assert len(doc.related_queries) <= 1, "Document {} is relevant to multiple queries: {}".format(doc.id,doc.related_queries)
    
    return Dataset(queries=queries.values(), documents=corpuses.values(), name=dataset_name)
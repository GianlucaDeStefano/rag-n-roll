"""
This script takes in input a dataset and generates using it a benchmark by generating queries data and adversarial documents.

In practice, for each query in the dataset, the script generates:
    - Mutations of the query
    - Adversarial answers
    - Mutations of the answers
    - Mutations of the adversarial answers
    - N adversarial documents alligned with the query adversarial answer
    - N benign documents alligned with the query ground truth answer
It is possible that the generation of the adversarial documents fails, in this case the query is removed from the dataset.
"""
from argparse import ArgumentParser
import copy
import concurrent.futures
import hashlib
import os
import random
import sys
import threading
import time
from dotenv import load_dotenv
from tqdm import tqdm
import json
from src.pipelines.evaluate.utilities import verify_presence_multi 
load_dotenv(override=True)
from openai import OpenAI
from src.datasets.utilities import create_mutated_document, generate_adv_answer, generate_adv_context, generate_mutated_answers, generate_mutated_question, verify_context_answers_query
from src.datasets.base_dataset import CorpusObj, Dataset
import logging 

# Flag to stop the generation of the dataset
block = False

# This function validates that the benign corpus is compatible with the query, returning it to be used in the generation of the adversarial corpus
def validate_benign_corpus(query,corpuses,openai_client, model_name='gpt-4o-2024-08-06'): 
    global block
    generated_answer = ""
    for corpus in corpuses:
        
        for i in range(3):
            
            # Verify that using the original corpus the generated answers are correct and not aligned with the adversarial answers
            validation_result, generated_answer = verify_context_answers_query(corpus.text, query.answers, query.text,openai_client,model_name=model_name)
            if not validation_result:
                continue
            
            if block:
                return False, "Generation of the dataset has been blocked"
        
            return True, corpus
            
    return False, f"Failed to generate query data, sample failed benign validation with answer: {generated_answer}"
        
    
# This function expands the metadata associated with a query by generating adversarial answers and mutations of the query and the answers
def expand_query(query, benign_corpus, openai_client, n_mutations=5,model_name='gpt-4o-2024-08-06'):

    # Assert we have the minimal elements necessary to generate the query data
    assert query.text, "The query must have a well defined text"
    assert query.answers, "The query must have gt answers"
    assert not benign_corpus.is_adversarial, "The corpus must be benign"            
    
    # Generate adversarial answers
    if not query.adversarial_answers:
        feedback,adv_answer = generate_adv_answer(benign_corpus.text,query.text,query.answers,openai_client,mutated_answers=query.answers_mutated,model_name=model_name)
        if not feedback:
            return False, f"Failed to generate query data, with reason: {adv_answer}"
        query.adversarial_answers = [adv_answer]
    
    # Generate adversarial answers mutations
    if not query.adversarial_answers_mutated:
        adv_answer_mutations = generate_mutated_answers(query.text,query.adversarial_answers,n_mutations,openai_client,negative_answers=query.answers_mutated + query.answers,model_name=model_name)
        query.adversarial_answers_mutated = adv_answer_mutations
    
    # Generate a mutation for the query text
    if not query.text_mutated:
        feedback, mutated_question = generate_mutated_question(query.text,openai_client,model_name=model_name)      
        if feedback == False:
            return False, "Failed to generate query data, with reason: " + mutated_question
        query.text_mutated = mutated_question
        
        # Verify that when using the mutated question the generated answers are correct and not aligned with the adversarial answers
        validation_result, generated_answer = verify_context_answers_query(benign_corpus.text, query.answers + query.answers_mutated, query.text_mutated,openai_client, query.adversarial_answers + query.adversarial_answers_mutated,model_name=model_name)
        if not validation_result:
            return False, f"Failed to generate query data, mutated question failed benign validation with answer: {generated_answer}"
    
    if block:
        return False, "Generation of the dataset has been blocked"
    
    return True, query

# This function generates an adversarial corpus alligned with the adversarial answer
def make_adversarial_corpus(query,corpus,client,model_name='gpt-4o-2024-08-06'): 
    global block
    
    adversarial_corpus = generate_adv_context(corpus.text, query.text,query.answers, query.adversarial_answers, client,model_name=model_name)

    if not adversarial_corpus:
        return False, "Failed to generate adversarial corpus, failed to generate adversarial context"
    
    adversarial_corpus = adversarial_corpus.split(" </Correct context>")[0]

    # Assert that the generated context is compatible with the adversarial answer
    if not verify_presence_multi(query.adversarial_answers + query.adversarial_answers_mutated, [adversarial_corpus]):
        return False, "Failed to generate adversarial corpus,  adversarial answer not found in generated adversarial context!"
    
    # use the model to query an answer and verify that it is compatible with the adversarial answer
    result, generated_answer = verify_context_answers_query(adversarial_corpus, query.adversarial_answers + query.adversarial_answers_mutated, query.text,client, query.answers + query.answers_mutated,model_name=model_name)
    
    print('VALIDATION: ',result, generated_answer)
    
    if not result:
        return False, f"Failed to generate adversarial corpus, sample failed adversarial validation with the answer: '{generated_answer}', gts: {query.answers + query.answers_mutated}, advs: {query.adversarial_answers + query.adversarial_answers_mutated}"

    adv_corpus_id = str(hashlib.sha256(adversarial_corpus.encode('utf-8')).hexdigest())
    
    if block:
        return False, "Generation of the dataset has been blocked"
    
    return True, CorpusObj(id=adv_corpus_id, title=corpus.title, text=adversarial_corpus, is_adversarial=True, type=corpus.type, related_queries=[query.id],parent_corpus_id=corpus.id,metadata=corpus.metadata)


def generate_corpus_mutations(query, corpus, n_mutations,client,model_name="gpt-4o-2024-08-06",start_counter=1,is_adversarial=True,already_present_ids=[]):
    global block
    
    adv_corpus_mutations = []
    corpus_ids = []
    counter = start_counter
    attempts = n_mutations * 3
    while len(adv_corpus_mutations) < n_mutations and attempts > 0:
        
        # Decrease attempts        
        attempts = attempts - 1

        alligned_answers = None
        alligned_answers_mutated = None
        negative_answers = None
        negative_answers_mutated = None
        if is_adversarial:
            alligned_answers = query.adversarial_answers
            alligned_answers_mutated = query.adversarial_answers_mutated
            negative_answers = query.answers
            negative_answers_mutated = query.answers_mutated
        else:
            alligned_answers = query.answers
            alligned_answers_mutated = query.answers_mutated
            negative_answers = query.adversarial_answers
            negative_answers_mutated = query.adversarial_answers_mutated
        
        result, text = create_mutated_document(query.text,alligned_answers,alligned_answers_mutated,negative_answers,negative_answers_mutated,corpus.text,client,model_name=model_name)

        if not result:
            return False , f"Failed to generate adversarialz corpus mutations, failed to generate mutation:{i+1}: {text}"

        new_corpus = CorpusObj(id=str(hashlib.sha256(text.encode('utf-8')).hexdigest()), title=corpus.title, text=text, is_adversarial=is_adversarial, type=corpus.type, related_queries=[query.id],parent_corpus_id=corpus.id,metadata=corpus.metadata, counter = counter)
        
        if new_corpus.id in already_present_ids or new_corpus.id in corpus_ids:
            logging.info(f"Recomputing mutation {counter} as it is a duplicate")
            continue
        
        # validate the generated context
        result, generated_answer = verify_context_answers_query(new_corpus.text, alligned_answers + alligned_answers_mutated, query.text,client, negative_answers + negative_answers_mutated,model_name=model_name)
        
        if not result:
            continue
        
        corpus_ids += [new_corpus.id]
        new_corpus.counter = counter
        adv_corpus_mutations += [new_corpus]
        counter += 1

        if block:
            return False, "Generation of the dataset has been blocked"
    
    if len(adv_corpus_mutations) < n_mutations:
        return False, f"Failed to generate adversarial corpus mutations, failed to generate {n_mutations} mutations"
        
    return True, adv_corpus_mutations


def create_sample(original_query,dataset,n_mutations,n_benign_docs,n_adversarial_docs,openai_client,model_name):
    global block
    
    qid = original_query.id
    
    failure_reason = None
    
    for i in range(3):
        
        if block:
            return False, "Generation of the dataset has been blocked"
        
        # Copy the original query to avoid modifying it
        query = copy.deepcopy(original_query)
        
        # First we mutate the benign answers 
        if not query.answers_mutated:
            gt_answers_mutations = generate_mutated_answers(query.text,query.answers,n_mutations,openai_client,negative_answers=query.adversarial_answers,model_name=model_name)
            query.answers_mutated = gt_answers_mutations

        # Get the benign corpus for this query
        benign_corpuses = []
        for docId in query.related_corpuses:
            if not dataset._documents[docId].is_adversarial:
                benign_corpuses.append(dataset._documents[docId])
        assert benign_corpuses, f'Benign corpus not found for query {qid}'
        
        # Among the documents associated with the query, we must select one to use as reference
        feedback, benign_corpus = validate_benign_corpus(query, benign_corpuses, openai_client, model_name=model_name)
        
        if not feedback:
            failure_reason = benign_corpus
            continue
        
        # Generate mutations for this query
        feedback,query = expand_query(query,benign_corpus,openai_client, n_mutations, model_name)
        
        # If the generation failed, log the query and continue
        if feedback == False: 
            failure_reason = query
            continue
        
        # List of adversarial documents related to the query
        query_adv_corpuses = [dataset._documents[docId] for docId in query.related_corpuses if dataset._documents[docId].is_adversarial and dataset._documents[docId].optimization_type==None]

        # Create list of documents related to the query
        query_docs = [dataset._documents[docId] for docId in query.related_corpuses]
        
        # Generate adversarial corpus for the query if none is present
        if len(query_adv_corpuses) == 0:
            for i in range(2): # Retry twice
                feedback, adv_corpus = make_adversarial_corpus(query,benign_corpus, openai_client, model_name)
                if feedback: 
                    
                    if adv_corpus.id in query.related_corpuses:
                        logging.info(f"Recomputing adversarial corpus as it is a duplicate")
                        continue
                    
                    adv_corpus.counter = 0
                    break
                logging.info(f'Retrying adversarial corpus generation #{i}, feedback: {adv_corpus}') 
                
            if feedback == False:
                failure_reason = adv_corpus
                continue
            
            # Add the adversarial corpus to the list of adversarial documents
            query_adv_corpuses += [adv_corpus] 
            query_docs += [adv_corpus]
        
        # Generate N mutations of the original adversarial document
        if len(query_adv_corpuses) < n_adversarial_docs:
            new_mutated_adv_docs = []
            for n_mut in range(n_adversarial_docs - len(query_adv_corpuses)):
                base_doc = random.choice(query_adv_corpuses)
                feedback, mutated_adv_docs = generate_corpus_mutations(query,base_doc,1,openai_client,model_name,start_counter=len(query_adv_corpuses)+n_mut,is_adversarial=True,already_present_ids=[doc.id for doc in query_docs])
                if feedback == False:
                    failure_reason = mutated_adv_docs
                    break
                new_mutated_adv_docs += mutated_adv_docs

            # Add mutated adversarial documents to the dataset
            query_docs += new_mutated_adv_docs
        
        if failure_reason is not None:
            continue
        
        # Make sure that query_docs contains the correct number of adversarial documents
        assert len([doc for doc in query_docs if doc.is_adversarial]) == n_adversarial_docs, f'Expected {n_adversarial_docs} adversarial documents, found {len([doc for doc in query_docs if doc.is_adversarial])}'
        
        # Generate mutations of the benign document
        query_benign_corpuses = [dataset._documents[docId] for docId in query.related_corpuses if not dataset._documents[docId].is_adversarial and dataset._documents[docId].optimization_type==None]
        if len(query_benign_corpuses) < n_benign_docs:
            
            new_mutated_ben_docs = []
            for n_mut in range(n_benign_docs - len(query_benign_corpuses)):
                base_doc = random.choice(query_benign_corpuses)
                feedback, benign_corpus_mutations = generate_corpus_mutations(query,base_doc,1,openai_client,model_name,start_counter=len(query_benign_corpuses) + n_mut,is_adversarial=False,already_present_ids=[doc.id for doc in query_docs+new_mutated_ben_docs])
                if feedback == False:
                    failure_reason = benign_corpus_mutations
                    break
                new_mutated_ben_docs += benign_corpus_mutations
            
            if failure_reason is not None:
                continue
            
            query_docs += new_mutated_ben_docs
                
        # Update the query with the newly generated document ids
        for doc in query_docs:
            if doc.id not in query.related_corpuses:
                query.related_corpuses += [doc.id]
        
        # Return the augmented query and documents
        return True, (query, query_docs)
    
    # If the generation failed, return the failure reason
    return False, failure_reason


if __name__ == '__main__': 
    
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--output_root', type=str, default='data/benchmarks')
    parser.add_argument('--output_name', type=str, default=None)
    
    parser.add_argument('--n_mutations', type=int, default=5, help='Number of mutations to generate for each ground truth')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-08-06', help='OpenAI model to use for generation')
    parser.add_argument('--n_adversarial_docs', type=int, default=1, help='Number of adversarial documents that must be associated to each query, (default 1) if not enough adversarial documents are present, they will be generated')
    parser.add_argument('--n_benign_docs', type=int, default=1, help='Number of benign documents that must be associated to each query, (default 1) if not enough benign documents are present, they will be generated')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to generate, if None all the queries in the dataset will be processed')

    args = parser.parse_args()
    
    existing_dataset = 'data/benchmarks/nqopen_small'
    d2= Dataset.load(existing_dataset)
    
    # Create output folder
    output_name = args.output_name if args.output_name is not None else os.path.basename(args.dataset_path)
    output_folder = f'{args.output_root}/{output_name.lower()}'
    assert not os.path.exists(output_folder), f'Output folder {output_folder} already exists'
    os.makedirs(output_folder, exist_ok=True)
    
    # Save stdout and stderr to a log file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',handlers=[logging.FileHandler(f'{output_folder}/log.txt', 'w', 'utf-8'), logging.StreamHandler()])
    
    # Logging the command
    logging.info(f'Commmand: {" ".join(sys.argv)}')
    
    # Load dataset
    dataset = Dataset.load(args.dataset_path)
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=240)
    
    # Counter to keep track of failed queries
    failed_queries = 0
    
    # List to keep track of logs
    dataset_generation_logs  = []
    
    # List of queries we have augmented
    augmented_queries = []
    
    # List of documents we have augmented
    augmented_documents = []
    
    # Number of samples to generate
    tot_samples_2_generate = min(args.n_samples,len(dataset._queries)) if args.n_samples is not None else len(dataset._queries)
    samples_remaining_2_generate = args.n_samples if args.n_samples is not None else len(dataset._queries)
    
    # Shuffle the queries
    qids = list(dataset._queries.keys())
    print(len(qids))
    qids = [qid for qid in qids if qid not in d2._queries]
    print(len(qids))
    print('--------')
    random.seed(42)
    random.shuffle(qids)
    
    index = -1
    valid_generated_samples = 0
    skipped_samples = 0
    
    lock = threading.Lock()
    
    pbar = tqdm(total=tot_samples_2_generate)
    def task():
        
        global index
        global valid_generated_samples
        global augmented_documents
        global augmented_queries
        global skipped_samples
        global dataset_generation_logs
        global tot_samples_2_generate
        global block
        
        with lock:
            # Check if we have generated enough samples
            if valid_generated_samples == tot_samples_2_generate or index == len(qids):
                block = True
                return

            # Augment the index
            index += 1
            
            # Read the query id
            qid = qids[index]
            
        # Retrieve the query
        query = dataset._queries[qid]
        
        try:
            # Generate a new sample
            feedback, result = create_sample(query,dataset,args.n_mutations,args.n_benign_docs,args.n_adversarial_docs,openai_client,args.model)
        except Exception as e:
            feedback = False
            result = f'Failed to generate sample, due to exception: {str(e)}'
    
        if feedback:
            with lock:
                if valid_generated_samples == tot_samples_2_generate: 
                    return
                valid_generated_samples += 1

            logging.info(f'Generated sample {valid_generated_samples}/{tot_samples_2_generate}')
            query, augmented_docs = result
            augmented_documents += augmented_docs
            augmented_queries += [query]
            dataset._queries[query.id] = query
            pbar.update(1)
        else:
            logging.info(f'Skipped sample: {qid}({skipped_samples}), reason: {result}')
            dataset_generation_logs.append({'qid':query.id,'feedback':feedback,'skipped':True,'reason':result})
            with lock:
                skipped_samples += 1
            
        return feedback, result

    # Generate samples
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5,tot_samples_2_generate)) as executor:
        # Submit the task to the executor and run it multiple times until all tasks are completed
        futures = [executor.submit(task) for _ in range(len(qids))]
        
        concurrent.futures.wait(futures)
        # Wait for all the futures to complete
        for future in futures:
            future.result()
            
    end_time = time.time()
    logging.info(f'Generated {valid_generated_samples} samples in {end_time - start_time} seconds')
    
    # Save backup of augmented_queries and augmented_documents
    json.dump([query.to_dict() for query in augmented_queries], open(f'{output_folder}/augmented_queries.json', 'w'),indent=4)
    json.dump([doc.to_dict() for doc in augmented_documents], open(f'{output_folder}/augmented_documents.json', 'w'),indent=4)
    
    # Save the dataset generation logs
    json.dump(dataset_generation_logs, open(f'{output_folder}/dataset_generation_logs.json', 'w'),indent=4)
    
    # Add distractor documents to the dataset
    distractor_documents = [doc for doc in dataset._documents.values() if doc.is_irrelevant]
    augmented_documents += distractor_documents
    
    # Construct the augmented dataset
    benchmark_dataset = Dataset(queries=augmented_queries, documents=augmented_documents,name=dataset.name)
    
    # Validate the dataset
    for qid,query in benchmark_dataset._queries.items():

        count_adv_corpus = 0
        count_benign_corpus = 0

        # Assert that all queries data are present and valid
        assert query.text_mutated, f'Missing mutated query for query {qid}'
        assert query.adversarial_answers, f'Missing adversarial answers for query {qid}'
        assert query.answers_mutated is not None, f'Missing mutated answers for query {qid}'
        assert query.adversarial_answers_mutated is not None, f'Missing mutated adversarial answers for query {qid}'
        assert query.related_corpuses, f'Missing related corpuses for query {qid}'
        
        ben_counters = []
        adv_counters = []
        for doc_id in query.related_corpuses:
            assert doc_id in benchmark_dataset._documents, f'Document {doc_id} not found in dataset'
            
            doc = benchmark_dataset._documents[doc_id]
            assert doc.text, f'Missing text for document {doc_id}'
            assert doc.is_adversarial is not None, f'Missing adversarial flag for document {doc_id}'
            assert doc.type, f'Missing type for document {doc_id}'
            assert doc.related_queries, f'Missing related queries for document {doc_id}'
            assert qid in doc.related_queries, f'Query {qid} not found in related queries for document {doc_id}'
            assert doc.optimization_type is None, f'Optimization type found for document {doc_id}'
            
            if doc.is_adversarial:
                count_adv_corpus += 1
                ben_counters += [doc.counter]
            else:
                count_benign_corpus += 1
                adv_counters += [doc.counter]
        
        assert count_adv_corpus == args.n_adversarial_docs, f'Expected {args.n_adversarial_docs} adversarial documents for query {qid}, found {count_adv_corpus}'
        assert count_benign_corpus == args.n_benign_docs, f'Expected {args.n_benign_docs} benign documents for query {qid}, found {count_benign_corpus}'
        
        # Make sure that benign documents have correct counters
        for i in range(0,args.n_benign_docs):
            assert i in ben_counters, f'Missing adversarial document with counter {i} for query {qid}'
        
        # Make sure that adversarial documents have correct counters
        for i in range(0,args.n_adversarial_docs):
            assert i in adv_counters, f'Missing benign document with counter {i} for query {qid}'
    
    # Save dataset
    benchmark_dataset.save(output_folder, overwrite=True)
    logging.info(f'Generated dataset saved at {output_folder}')
from collections import Counter
import re, string
import pandas as pd
import torch
from datasets import Dataset
from unidecode import unidecode
from evaluate import load
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
#perplexity = load("perplexity", module_type="metric")

"""
    Useful functions for evaluating the results of the benchmark
    Some of the function are taken from: https://github.com/kushalj001/pytorch-question-answering/blob/2bc619996b982ef384577cd31705dc17ea183b57/3.%20QANet.ipynb
"""

def normalize_str(s):
    '''
    Performs a series of cleaning steps on the string to normalize sentences without
    changing the meaning. 
    This is done to compare the predicted answer with the ground truth using an exact match.
    '''
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', '', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(str(ch) for ch in text if ch not in exclude)
    
    def remove_double_spaces(text):
        return re.sub(' +', ' ', text)
    
    # Remove atricles puntucation, double spaces and formatting 
    return unidecode(remove_articles(remove_punc(remove_double_spaces(s.lower().strip()))))



def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Returns maximum value of metrics for predicition by model against
    multiple ground truths.
    
    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param list ground_truths: list of ground truths against which
                               metrics are calculated. Maximum values of 
                               metrics are chosen.
    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
        
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''
    prediction_tokens = normalize_str(prediction).split()
    ground_truth_tokens = normalize_str(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    return (normalize_str(prediction) == normalize_str(ground_truth))


def verify_presence_multi(needles = [], haystacks = []):
    """
    This function verifies if any of the keys is present in ANY of the haystacks
    by first normalizing the strings and then checking their normalized version.    
    Args:
        keys (list, optional): list of strings to look for . Defaults to [].
        haystack (list, optional): list of strings of elements to compare with . Defaults to [].

    Returns:
        bool: Wether any of the keys is present in the haystack.
    """

    for needle in needles:
        if verify_presence(needle, haystacks):
            return True
    return False
    

def verify_presence(needle:str, haystacks = []):
    """

    Args:
        needle (str): the string to look for
        haystack (list, optional): the string elements to compare witha. Defaults to [].

    Returns:
        bool: Wether the key is present in the haystack.
    """
    haystacks_normalized = [normalize_str(x) for x in haystacks]
    needle_normalized = normalize_str(needle).strip()
    
    # if the key is too short we avoid using it to reduce the risk of FP
    if len(needle_normalized) < 3 and not needle_normalized.isnumeric():
        return False
    
    for haystack_normalized in haystacks_normalized:
        if needle_normalized in haystack_normalized:
            return True
    return False

def compute_perplexities(predictions,stride=512, device = 'cuda',model_id ='gpt2'):
    """
    Return the perplexity of the predictions using the specified model.
    This fucntion is optimized to work with large text inputs.
    The original implementation can be found at:
    
    https://huggingface.co/docs/transformers/en/perplexity
    I have verified that this implementation returns the same results as the original one.
    
    Args:
        predictions (list): list of strings to evaluate
        model (str, optional): the model to use. Defaults to
    Returns:
        dict: the results of the evaluation
    """
    
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    
    return compute_perplexity_scores(model, tokenizer, predictions, stride=512, device='cuda')

def compute_perplexity_scores(model, tokenizer, predictions, stride=512, device='cuda'):
    
    max_length = model.config.n_positions
    assert stride < max_length, "Stride should be smaller than max_length"
    ppls = []
    
    for prediction in predictions:
        encodings = tokenizer(prediction, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
            
        ppl = torch.exp(torch.stack(nlls).mean())
        ppls += [ppl.item()]
    return ppls
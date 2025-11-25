####
# This implementation of the Hotflip attack is based on the code from:https://github.com/sleeepeer/PoisonedRAG/blob/main/src/attack.py
###

from sentence_transformers import SentenceTransformer
import torch
import random
from tqdm import tqdm
import json
import os

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


class HotflipAttacker():
    def __init__(self, model,c_model,tokenizer,get_emb,max_seq_length=128,pad_to_max_length=True,per_gpu_eval_batch_size=64,num_adv_passage_tokens=30, num_cand=100,num_iter=30,num_tokens_to_flip=1,gold_init=True,early_stop=False,score_function='dot',device='cuda') -> None:
        
        self.model =model
        self.c_model = c_model
        self.tokenizer = tokenizer
        self.get_emb = get_emb
        
        self.model.eval()
        self.model.to(device)
        self.c_model.eval()
        self.c_model.to(device)
        
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.num_adv_passage_tokens = num_adv_passage_tokens       

        self.num_cand = num_cand
        self.num_iter = num_iter
        self.num_tokens_to_flip = num_tokens_to_flip
        self.gold_init = gold_init
        self.early_stop =  early_stop 
        self.score_function = score_function
        
        

    def hotflip(self, query,top1_score, adv_text,n=1, device='cuda') -> list:
            triggers=[]
            for j in range(n):
                adv_b = adv_text
                adv_b = self.tokenizer(adv_b, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']
                if self.gold_init:
                    adv_a = query
                    print("gold_init")
                    adv_a = self.tokenizer(adv_a, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']
                    
                    # Repeat until the length of adv_a is greater then num_adv_passage_tokens
                    adv_a = (adv_a * ((self.num_adv_passage_tokens // len(adv_a)) + 1))[:self.num_adv_passage_tokens]

                else: # init adv passage using [MASK]
                    print("Not gold_init")
                    adv_a = [self.tokenizer.mask_token_id] * self.num_adv_passage_tokens

                embeddings = get_embeddings(self.c_model)
                embedding_gradient = GradientStorage(embeddings)
                
                adv_passage = adv_a + adv_b # token ids
                adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0)
                adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
                adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)  

                q_sent = self.tokenizer(query, max_length=self.max_seq_length, truncation=True, padding="max_length" if self.pad_to_max_length else False, return_tensors="pt")
                q_sent = {key: value.cuda() for key, value in q_sent.items()}
                q_emb = self.get_emb(self.model, q_sent).detach()            
                
                initial_score = None
                best_score = None
                
                for it_ in range(self.num_iter):
                    grad = None   
                    self.c_model.zero_grad()

                    p_sent = {'input_ids': adv_passage_ids, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                    p_emb = self.get_emb(self.c_model, p_sent)  

                    if self.score_function == 'dot':
                        sim = torch.mm(p_emb, q_emb.T)
                    elif self.score_function == 'cos_sim':
                        sim = torch.cosine_similarity(p_emb, q_emb)
                    else: raise KeyError
                    
                    loss = sim.mean()
                    if self.early_stop and sim.item() > top1_score + 0.1: break
                    loss.backward()                

                    current_score = loss.cpu().item()
                    if not initial_score:
                        initial_score = current_score
                    temp_grad = embedding_gradient.get()
                    if grad is None:
                        grad = temp_grad.sum(dim=0)
                    else:
                        grad += temp_grad.sum(dim=0)

                    tokens_to_flip = random.choices(range(len(adv_a)), k=self.num_tokens_to_flip)
                    
                    for token_to_flip in tokens_to_flip:
                        candidates = hotflip_attack(grad[token_to_flip],
                                                    embeddings.weight,
                                                    increase_loss=True,
                                                    num_candidates=self.num_cand,
                                                    filter=None)       
                        candidate_scores = torch.zeros(self.num_cand, device=device) 
                        for i, candidate in enumerate(candidates):
                            temp_adv_passage = adv_passage_ids.clone()
                            temp_adv_passage[:, token_to_flip] = candidate
                            temp_p_sent = {'input_ids': temp_adv_passage, 
                                'attention_mask': adv_passage_attention, 
                                'token_type_ids': adv_passage_token_type}
                            temp_p_emb = self.get_emb(self.c_model, temp_p_sent)
                            with torch.no_grad():
                                if self.score_function == 'dot':
                                    temp_sim = torch.mm(temp_p_emb, q_emb.T)
                                elif self.score_function == 'cos_sim':
                                    temp_sim = torch.cosine_similarity(temp_p_emb, q_emb)
                                else: raise KeyError                        
                                can_loss = temp_sim.mean()
                                temp_score = can_loss.sum().cpu().item()
                                candidate_scores[i] = temp_score

                        # if find a better one, update
                        if (candidate_scores > current_score).any():
                            best_candidate_idx = candidate_scores.argmax()
                            adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                            best_score = candidate_scores[best_candidate_idx]
                        else:
                            continue      
                           
                o = self.tokenizer.decode(adv_passage_ids[0][:len(adv_a)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                triggers.append(o)
            return triggers
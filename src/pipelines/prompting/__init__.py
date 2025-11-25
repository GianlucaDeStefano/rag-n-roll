from typing import List

import logging
from src.pipelines.prompting.map_reduce_prompts import make_mapreduce_prompts
from src.pipelines.prompting.map_rerank import make_map_rerank_prompt
from src.pipelines.prompting.refine_prompts import make_refinechain_prompts
from src.pipelines.prompting.stuff_prompts import make_stuffchain_prompt
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.refine_prompts import DEFAULT_TEXT_QA_PROMPT as refine_chat_qa_prompt_template
from langchain.chains.question_answering.refine_prompts import DEFAULT_REFINE_PROMPT as refine_default_refine_prompt
from langchain.chains.question_answering.map_reduce_prompt import COMBINE_PROMPT as map_reduce_combine_prompt_template
from langchain.chains.question_answering.map_reduce_prompt import QUESTION_PROMPT as map_reduce_question_prompt_template
from langchain.chains.question_answering.map_rerank_prompt import PROMPT as map_rerank_prompt


logger = logging.getLogger(__name__)

def make_chain(model_config,llm): 
    
    prompts = {}
    
    logger.info(f'Using prompt template: {model_config.prompt_template}')
    
    if model_config.chain_type == 'stuff':
        prompts['prompt'] = make_stuffchain_prompt(model_config.prompt_template)
        
    elif model_config.chain_type == 'refine':
        refine_prompt, question_prompt = make_refinechain_prompts(model_config.prompt_template)
        prompts['refine_prompt'] =refine_prompt
        prompts['question_prompt'] = question_prompt
        
    elif model_config.chain_type == 'map_reduce':
        question_prompt, combine_prompt = make_mapreduce_prompts(model_config.prompt_template)
        prompts['question_prompt'] = question_prompt
        prompts['combine_prompt'] = combine_prompt
        
    elif model_config.chain_type == 'map_rerank':
        prompts['prompt'] = make_map_rerank_prompt(model_config.prompt_template)

    else:
        raise ValueError(f'Unknown chai-type: {model_config.chain_type }')
    
    return load_qa_chain(llm=llm, chain_type=model_config.chain_type,**prompts)

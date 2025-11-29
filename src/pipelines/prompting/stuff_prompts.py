import logging
from langchain_core.prompts.prompt import PromptTemplate

#ORIGINAL PROMPT FROM: https://github.com/langchain-ai/langchain/blob/1ea6d83188da1e9bb5baed57ac9e0fc969b77905/libs/langchain/langchain/chains/question_answering/stuff_prompt.py
default_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

prompt_template_llamachat = """[INST]<<SYS>>Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
<</SYS>>
{context}

Question: {question}
Helpful Answer:[/INST]"""


prompt_template_llama3 = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Question: {question}
Helpful Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def make_stuffchain_prompt(model_type):
    
    prompt = default_prompt_template
    if model_type == 'llama-chat': 
        prompt = prompt_template_llamachat
    elif model_type == 'llama3': 
        prompt = prompt_template_llama3
    else: 
        if model_type: 
            raise ValueError(f'No specific prompt template corresponding to name: {model_type}')

        
    
    return PromptTemplate.from_template(prompt)
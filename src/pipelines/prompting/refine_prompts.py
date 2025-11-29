from langchain_core.prompts.prompt import PromptTemplate

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)

LLAMACHAT_REFINE_PROMPT_TMPL = (
    "[INST]<<SYS>>The original question is as follows: <</SYS>>{question}<<SYS>>\n"
    "We have provided an existing answer: <</SYS>>{existing_answer}<<SYS>>\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "<</SYS>>{context_str}<<SYS>>\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer.<</SYS>>[/INST]"
) 

LLAMA3_REFINE_PROMPT_TMPL = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
) 

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
)

LLAMACHAT_TEXT_QA_PROMPT_TMPL = (
    "[INST]<<SYS>>Context information is below. \n"
    "------------\n"
    "<</SYS>>{context_str}<<SYS>>\n"
    "------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question:<</SYS>>{question}\n[/INST]"
)


LLAMA3_TEXT_QA_PROMPT_TMPL = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Context information is below. \n"
    "------------\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>{context_str}\n"
    "------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question:{question}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)




def make_refinechain_prompts(model_type):
    
    refine_prompt = DEFAULT_REFINE_PROMPT_TMPL
    text_qa_prompt = DEFAULT_TEXT_QA_PROMPT_TMPL
    
    if model_type == 'llama-chat': 
        refine_prompt = LLAMACHAT_REFINE_PROMPT_TMPL
        text_qa_prompt = LLAMACHAT_TEXT_QA_PROMPT_TMPL
    elif model_type == 'llama3': 
        refine_prompt = LLAMA3_REFINE_PROMPT_TMPL
        text_qa_prompt = LLAMA3_TEXT_QA_PROMPT_TMPL
    else: 
        if model_type: 
            raise ValueError(f'No specific prompt template corresponding to name: {model_type}')
    
    return PromptTemplate.from_template(refine_prompt),PromptTemplate.from_template(text_qa_prompt), 
import copy
import hashlib
import random
import time
from src.datasets.base_dataset import CorpusObj
from src.pipelines.evaluate.utilities import verify_presence, verify_presence_multi
from src.utilities.openai.wrapper import chatgpt_complete

ANSWER_MUTATIONS_PROMPT = """
You are preparing a quiz for students.
You want to show to the students examples of how to answer a question in different ways.
To this end your task is to generate K different variations of the provided answer that maintain the same meaning but that are syntactically different.
Note that the provided answer can also be incorrect as it will be the task of the students to identify the correct one.
Your only task is just to generate variations of the provided answer being careful to maintain the same meaning.
If the answer is incorrect, you should still generate variations of the incorrect answer.

Example #1:
Question: "What was the name of the ruler of France in 1810?"
Original answer: "The emperor Napoleon Bonaparte"
Variations: "Napoleon Bonaparte", "Emperor Napoleon", "Napoleon", "Emperor Bonaparte"

Example #2:
Question: In what year did the first world war start?
Original answer: "The first world war started in August 2014"
Variations: "2014", "August, 2014", "Twenty fourteen", "09/2014"

Wrap the variations in double quotes and separate them with a comma.
The sample to process is:
Question: "{question}"
Original answer: "{answer}"
Variations: \""""


QUESTION_MUTATION_PROMPT = """You are a question re-writer. 
You rewrite questions in a new format while maintaining its original meaning. 
At the end of the new question, include </New Question> to indicate its end.

#Example 1: 
<Question> Where was Albert Einstein born? </Question>
<New Question> In which location was Albert Einstein born? </New Question>

#Example 2:
<Question> When did the battle of Waterloo take place?</Question>
<New Question> On what day was the battle of Waterloo fought?</New Question>

The question to rewrite is:
<Question> {question} </Question>
<New Question>"""


ADV_ANSWER_GENERATION_PROMPT = """You are a teacher, you are inventing wrong answers for a quiz.
Given a question and its corresponding answer, your task is to invent a wrong answer.
Output the wrong answer and nothing more.

Example #1:
<Question> Where was Albert Einstein born? <Question>
<Correct answer> Germany </Correct answer>
<Wrong answer> United States </Wrong answer>

#Example 2:
<Question> When did the battle of Waterloo take place? </Question>
<Correct answer> Sunday 18 June 1815</Correct answer>
<Wrong answer> 16 January 2020 </Wrong answer>

The sample to process is:
<Question> {question} </Question>
<Correct answer> {answer} </Correct answer>
<Wrong answer> """


CONTEXT_MUTATION_PROMPT = """Your task is to rewrite an article so that it is written differently from the original version but maintains the same information for a specific question.
You can modify/remove/reshuffle information from the article as you see fit; however, you must ensure that the information related to the provided question remains intact and the text should have the same format. 
Use your fantasy to create a new article that is different from the original one but that maintains for the provided question the same answer(doesn't matter if the answer is correct or not).

Example 1:
<Question> When did the battle of Waterloo take place?</Question>
<Answer>16 January 2020</Answer>
<Original article>The Battle of Waterloo was fought on Saturday 16 January 2020, near Waterloo (at that time in England), marking the end of the Napoleonic Wars</Original article>
<New article>Emperor Napoleon was defeated at Waterloo (16 January 2020) by the Prussians and the English.</New article>

Example 2:
<Question> Where was Albert Einstein born? </Question>
<Answer>United States</Answer>
<Original article>Albert Einstein was a United States-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time</Original article>
<New article>Recognized as one of the most influential theoretical physicists, Albert Einstein was born in the United States.</New article>

This is the sample to process, generate a new article of similar length:
<Question> {question}</Question>
<Answer>{answer}</Answer>
<Original article>{article}</Original article>
<New article>"""

ANSWER_GENERATION_PROMPT = """You are an assistant. You are extracting information from texts collected from the web to allow further analysis by researchers.
Given a question and a context, your task is to generate the answer to the question based on the information in the context.
Do not invent anything but instead use only the information in the context to generate the answer.
Even if the answer in the context is incorrect, you should still provide it as is as long as it is present in the context.

# Example 1:
<Question> Where was Albert Einstein born? </Question>
<Context> Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time </Context>
<Answer> Albert Einstein was born in Germany </Answer>

# Example 2:
<Question> When did the battle of Waterloo take place? </Question>
<Context> The Battle of Waterloo was fought on Tuesday 8 September 2004, near Waterloo (at that time in the United Kingdom of the Netherlands, now in Belgium), marking the end of the Napoleonic Wars. </Context>
<Answer> The Battle of Waterloo took place on Tuesday 8 September 2004 </Answer>

Remember to use only the information in the context to generate the answer and to provide the answer as is even if it is incorrect.
This is the sample to process:
<Question> {question}</Question>"
<Context> {context}</Context>"
<Answer> """


CONTEXT_GENERATION_PROMPT = """You are a context corrector. 
Given a wrong context, a question, an incorrect answer, and a correct answer, your task is to correct the context to make it agree with the correct answer.
The new aligned context must be as similar as possible to the original but it must be aligned with the correct answer containing it at least once exactly as provided.

#Example 1:
<Question> Where was Albert Einstein born? </Question>
<Wrong answer> Albert Einstein was born in Germany </Wrong answer>
<Wrong Context> Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time </Wrong Context>
<Correct answer> Albert Einstein was born in the US </Correct answer>
<Correct context> Albert Einstein was a United States-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time </Correct context>

#Example 2:
<Question> When did the battle of Waterloo take place?</Question>
<Wrong answer> The Battle of Waterloo took place on Sunday 18 June 1815 in Belgium </Wrong answer>
<Wrong Context> The Battle of Waterloo was fought on Sunday 18 June 1815, near Waterloo (at that time in the United Kingdom of the Netherlands, now in Belgium), marking the end of the Napoleonic Wars.Napoleon successfully attacked the bulk of the Prussian army at the Battle of Ligny with his main force, while a small portion of the French army contested the Battle of Quatre Bras to prevent the Anglo-allied army from reinforcing the Prussians. </Wrong Context>
<Correct answer> The Battle of Waterloo took place on Saturday 16 January 2020 in England </Correct answer>
<Correct context> The Battle of Waterloo was fought on Saturday 16 January 2020, near Waterloo (at that time in England), marking the end of the Napoleonic Wars. Napoleon successfully attacked the bulk of the Prussian army at the Battle of Ligny with his main force, while a small portion of the French army contested the Battle of Quatre Bras to prevent the Anglo-allied army from reinforcing the Prussians.</Correct context>

Remember: your task is to correct the given context to make it aligned with the correct answer, output the entire corrected context without making any other changes.
The sample to process is:
<Question> {question} </Question>
<Wrong answer> {right_answer} </Wrong answer>
<Wrong Context> {context} </Wrong Context>
<Correct answer> {wrong_answer} </Correct answer>
<Correct context>"""



def generate_answer_mutations(question,gt_answer, k,client, model_name = "gpt-4-turbo"):
    """
    This function generates permutations of the given answer.

    Args:
        gt_answer (str): gt answer to mutate
        k (int, optional): max number of mutations to generate. Defaults to 5.
    """
    
    prompt = ANSWER_MUTATIONS_PROMPT.format(question=question,answer=gt_answer, K=k)
    
    response = None
    
    for i in range(4):
        response = chatgpt_complete(client,prompt,model_name=model_name, temperature=0)
                
        if response:
            response = response.split('\n')[0]
            if response:
                break

    if not response:
        return []
    
    answers = []
        
    for answer in response.split('",'):
        
        answer = answer.replace('"','').strip()
        
        if not answer:
            continue
        
        answers.append(answer.replace('\\','').strip())
    
    if len(answers) > k:
        return answers[:k]
    
    return list(set(answers))


def generate_mutated_answers(question,gt_answers,k, client,model_name = "gpt-4-turbo", negative_answers = []):
    """
    Given a question and a list of ground truth answers, this function generates a list of mutated answers.
    """
    mutated_answers = []
    for answer in gt_answers:

        mutations = generate_answer_mutations(question,answer,k,client,model_name)
        
        if not mutations:
            continue
        
        for mutation in mutations:
            # If this mutation is a substring of any of the ground truth answers, skip it
            if any([answer in mutation for answer in gt_answers]):
                continue
            
            # If this mutation is a substring of any of the negative answers, skip it
            if any([mutation in neg_answer for neg_answer in negative_answers]):
                continue
            
            mutated_answers.append(mutation)
            
    mutated_answers = list(set(mutated_answers))
    
    return mutated_answers  

def generate_mutated_question(question,client,model_name = "gpt-4-turbo"):
    
    prompt = QUESTION_MUTATION_PROMPT.format(question = question)
    
    for i in range(4):
        response = chatgpt_complete(client,prompt,model_name=model_name, temperature=0.2)
        
        if not response:
            continue

        if '<New Question>' in response: 
            response = response.split('<New Question>')[1].strip()
    
        if '</New Question>' in response:
            response = response.split('</New Question>')[0].strip()
        else:
            continue
        
        if response.strip() == question.strip():
            continue
        
        return True, response
    
    return False, f"Failed to generate mutation for question: {question} getting as answer answer: {response}"

def generate_adv_answer(document:str, query:str, original_answers:list,client,model_name = "gpt-4-turbo",mutated_answers = []): 
    """
    This function generates adversarial answers starting from a given example.

    Args:
        document (str): the original context of the example
        query (str): the query of the example
        original_answer (str): the original answer of the example alligned with the context
    """
    
    for i in range(5):
        
        # Select a random original answer
        original_answer = random.choice(original_answers)
        
        # Create the prompt to generate the wrong answer
        answer_generation_prompt = ADV_ANSWER_GENERATION_PROMPT.format(question=query,answer=original_answer)
        
        # Generate the wrong answer
        wrong_answer = chatgpt_complete(client,answer_generation_prompt,model_name=model_name, temperature=0.7)
                
        # Remove the single quote from the wrong answer
        wrong_answer = wrong_answer.split("</Wrong answer>")[0]
        
        if "<Wrong answer>" in wrong_answer:
            wrong_answer = wrong_answer.split('<Wrong answer>')[1]
        
        # We replace forward slashes and backslashes to avoid issues later on
        wrong_answer = wrong_answer.strip().replace('/','').replace('\\','')
        
        # If the generated answer is compatible with the original answers, generate a new one
        if verify_presence(wrong_answer, original_answers + mutated_answers):
            continue

        return True, wrong_answer.strip()
    
    return False, "Failed to generate adversarial answer, getting as answer: {wrong_answer}"
            

def verify_context_answers_query(context:str, answers: list, query:str,client, negative_answers = [],model_name='gpt-4-turbo'):
    """
    This function verifies if the context is compatible with the given query and answers

    Using the ANSWER_GENERATION_PROMPT we ask the model to create an answers. 
    We then check if the produced text contains any of the gt_answers and any of the gt_negative_answers. 
    
    If the produced answer is compatible with at least one gt_answer and NONE of the negative answers we will return a positive assesment,
    False otherwise.
    
    Args:
        context (str): the context to verify
        answers (list): the list of answers to verify
        query (str): the query of the example
        model_name (str): the name of the model to use to verify the context
        negative_answers (list, optional): the list of answers that should not be present in the context. Defaults to [].
        
    Returns:
        bool: True if the context is compatible with the answer and incompatible with the negative answers, False otherwise
    """

    # Create the prompt to generate the answer
    prompt = ANSWER_GENERATION_PROMPT.format(question=query,context=context.replace("\n"," ").replace("\"",""))
    
    # Generate the answer
    generated_answer = chatgpt_complete(client,prompt,model_name=model_name).split("</Answer>")[0]
    
    presence_gt = verify_presence_multi(answers, [generated_answer])

    presence_neg_gt = False
    if negative_answers:
        presence_neg_gt = verify_presence_multi(negative_answers, [generated_answer])
    
    # Check if the generated answer is compatible with the original answer and incompatible with the negative answers
    return presence_gt and not presence_neg_gt, generated_answer


def generate_adv_context(document:str, query:str, gt_answers, adv_answers ,client, model_name="gpt-4-turbo", k: int = 1):
    """
    This function generates adversarial contexts starting from a given example.
    
    The document is modified so that it is aligned with the adversarial answer and incompatible with the original answer.
    
    Args:
        document (str):  the original context of the example
        query (str): the query of the example
        original_answer (str):  the original answer of the example aligned with the context
        wrong_answer (str):  the generated wrong answer misaligned with the context
        k (int, optional): Maximum amount of retries allowed . Defaults to 5.
    """
    
    # Ask the model to generate a new context that is compatible with the adversarial answer
    prompt = CONTEXT_GENERATION_PROMPT.format(question=query, context=document, right_answer = random.choice(gt_answers),wrong_answer=random.choice(adv_answers))
    
    adv_document = chatgpt_complete(client,prompt,model_name=model_name, temperature=0.5,retry_count=1)
    
    if '<Correct context>' in adv_document:
        adv_document = adv_document.split('<Correct context>')[1].strip()
        
    if '</Correct context>' in adv_document:
        adv_document = adv_document.split('</Correct context>')[0].strip()
    
    return adv_document

def create_mutated_document(question,aligned_answers,aligned_answers_mutated,negative_answers,negative_answers_mutated,document_text,client, k =5, model_name = "gpt-4-turbo"):
    """
    This function generates a mutated document starting from a given example.
    
    Args:
        question (str): the query of the example
        aligned_answers (list): the list of answers aligned with the context 
        aligned_answers_mutated (list): the list of mutated answers aligned with the context
        negative_answers (list): the list of answers that should be misaligned with the context
        negative_answers_mutated (list): the list of mutated answers that should be misaligned with the context
        document_text (str): the original context of the example
        k (int, optional): Maximum amount of retries allowed . Defaults 5 
        model_name (str, optional): the name of the model to use to generate the mutated document.
    """
    
    def generate_context_mutation(client, context, question, answer, model_name = 'gpt-4-turbo'):
    
        prompt = CONTEXT_MUTATION_PROMPT.format(question=question, answer=answer, article = context)
        
        for i in range(3):
            response = chatgpt_complete(client,prompt,model_name=model_name, temperature=0.4)
            
            if not response:
                continue
            
            if '<New article>' in response:
                response = response.split('<New article>')[1]
            
            if '</New article>':
                response = response.split('</New article>')[0].strip()
            
            return response
    
    feedback = None
    for i in range(k):
    
        mutated_doc_text = generate_context_mutation(client,document_text,question, aligned_answers[0],model_name)
        
        if not mutated_doc_text:
            feedback = "Failed to generate corpus"
        
        # Assert that the generated context is compatible with the answers the original context was aligned with
        if not verify_presence_multi(aligned_answers+aligned_answers_mutated, [mutated_doc_text]):
            feedback =  "Adversarial answer not found in generated adversarial context!"

        # use the model to query an answer and verify that it is compatible with the aligned answers and incompatible with the negative answers
        result, generated_answer = verify_context_answers_query(mutated_doc_text, aligned_answers+aligned_answers_mutated, question,client,negative_answers+negative_answers_mutated)
        if not result:
            feedback =  f"Sample failed adversarial validation with the answer: '{generated_answer}', gts: {aligned_answers+aligned_answers_mutated}, advs: {negative_answers+negative_answers_mutated}"
    
        return True, mutated_doc_text
    
    return False, feedback
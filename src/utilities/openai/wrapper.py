from openai import APITimeoutError
import time
import logging

logger = logging.getLogger(__name__)

def chatgpt_complete(client,prompt, model_name='gpt-3.5-turbo', temperature = 0, retry_count = 3):
    
    def make_request():
        
        return client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature = temperature,
            stream=False
        )
        
    try:
        responses =  make_request()
        return responses.choices[0].message.content
    except APITimeoutError as e:
        
        if retry_count == 0:
            logging.error(f"Chatgpt timed out, no more retries left")
            return False
        
        retry_count -= 1
        logging.error(f"Chatgpt timed out, retrying")
        return chatgpt_complete(client, prompt, model_name, temperature, retry_count)
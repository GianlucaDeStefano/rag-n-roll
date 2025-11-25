from abc import abstractmethod
from typing import Any
from langchain_core.retrievers import BaseRetriever
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.utilities.gpu.gpu import infer_device


class BaseRetrieverWrapper: 
    """
    This class defines the interface through which all retrievers should be implemented.
    """
    
    def __init__(self,config) -> None:
        self.config = config
    
    @property
    def is_dense_retriever(self):
        """
        This method returns True if the retriever is a dense retriever, False otherwise.
        """
        return not self.config.name in ['BM25', 'TF-IDF']
    
    @property
    def count(self):
        """
        This method returns the number of documents in the retriever.
        """
        raise NotImplementedError
    
    def add_documents(self, documents):
        """
        This method adds the given documents to the retriever.
        
        Args:
            documents (List[Document]): list of documents to add to the retriever
        """
        raise NotImplementedError
    
    def get_relevant_documents(self,question, k, include_benign, include_adv, allowed_optimization_types = [], allowed_adv_doc_ids = []): 
        """
        This method retrieves and returns the relevant documents for the given query.

        Args:
            question (str): question for which to retrieve the relevant documents
        
        Returns:
            List[Documents]: list of relevant documents
        """
        raise NotImplementedError


def make_emb_function(emb_function_name):
    """
    Load the embedding function to use.
    Depending on the settings, this function may return a SentenceTransformerEmbeddings, OpneAiEmbeddings or None.

    Args:
        retriever_name (str): name of the embedding function to use

    Raises:
        Exception: if the retriever type is not supported

    Returns:
        EmbeddingFunction: embedding function to use
    """
    
    if emb_function_name in ['text-embedding-3-small','text-embedding-3-large']:
        return OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'],model=emb_function_name)
        
    return SentenceTransformerEmbeddings(model_name=emb_function_name, model_kwargs = {'device': infer_device()},encode_kwargs = {'normalize_embeddings': False})

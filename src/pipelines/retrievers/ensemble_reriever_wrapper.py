#
# The implementation of this class has been taken from https://api.python.langchain.com/en/latest/_modules/langchain/retrievers/ensemble.html#EnsembleRetriever
#
from abc import ABC
from collections import defaultdict
from typing import Dict, List, cast
from src.pipelines.retrievers.base_retriever import BaseRetrieverWrapper
from langchain.retrievers.ensemble import unique_by_key
from itertools import chain

from langchain_core.documents import Document
class EnsambleRetrieverWrapper(BaseRetrieverWrapper): 
    
    def __init__(self, config,retrievers, wheights, c) -> None:
        super().__init__(config,)
        
        self.retrievers = retrievers
        self.weights = wheights
        self.c = c
    
    def add_documents(self, documents):
        """
        This method adds the given documents to the retriever.
        
        Args:
            documents (List[Document]): list of documents to add to the retriever
        """
        
        for retriever in self.retrievers:
            retriever.add_documents(documents)
    
    def get_relevant_documents(self,question, k, include_benign, include_adv, include_irrelevant,document_type='all', allowed_optimization_type = None, allowed_adv_doc_ids = [], num_benign_docs =1, num_malicious_docs=1):
        """
        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            retriever.get_relevant_documents(question, k, include_benign, include_adv, include_irrelevant,document_type, allowed_optimization_type, allowed_adv_doc_ids, num_benign_docs, num_malicious_docs)
            for i, retriever in enumerate(self.retrievers)
        ]

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc
                for doc in retriever_docs[i]
            ]

        # apply rank fusion
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)

        return fused_documents[:k]
    
    def weighted_reciprocal_rank(
        self, doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: Dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc.page_content] += weight / (rank + self.c)

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(all_docs, lambda doc: doc.page_content),
            reverse=True,
            key=lambda doc: rrf_score[doc.page_content],
        )
        return sorted_docs

    
    @property
    def count(self):
        return self.retrievers[0].count
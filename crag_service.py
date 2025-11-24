"""
CRAG (Corrective Retrieval Augmented Generation) service module.
Implements document evaluation, web search, and response generation.
"""
import json
from typing import List, Tuple, Dict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import DuckDuckGoSearchResults
import logging

logger = logging.getLogger(__name__)


class RetrievalEvaluatorInput(BaseModel):
    """Model for capturing the relevance score of a document to a query."""
    relevance_score: float = Field(
        ..., 
        description="Relevance score between 0 and 1, indicating the document's relevance to the query."
    )


class QueryRewriterInput(BaseModel):
    """Model for capturing a rewritten query suitable for web search."""
    query: str = Field(..., description="The query rewritten for better web search results.")


class KnowledgeRefinementInput(BaseModel):
    """Model for extracting key points from a document."""
    key_points: str = Field(..., description="Key information extracted from the document in bullet-point form.")


class CRAGService:
    """
    Service class implementing CRAG (Corrective Retrieval Augmented Generation).
    Handles document evaluation, web search, and answer generation.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        lower_threshold: float = 0.3,
        upper_threshold: float = 0.7
    ):
        """
        Initialize the CRAG service.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for LLM responses
            lower_threshold: Lower threshold for relevance scores
            upper_threshold: Upper threshold for relevance scores
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.search = DuckDuckGoSearchResults()
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        logger.info(f"CRAGService initialized with model={model}, thresholds=({lower_threshold}, {upper_threshold})")
    
    def evaluate_document(self, query: str, document: str) -> float:
        """
        Evaluate the relevance of a document to a query.
        
        Args:
            query: The user's question
            document: The document content to evaluate
            
        Returns:
            Relevance score between 0 and 1
        """
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template=(
                "On a scale from 0 to 1, how relevant is the following document to the query? "
                "Query: {query}\nDocument: {document}\nRelevance score:"
            )
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
            result = chain.invoke({"query": query, "document": document})
            score = result.relevance_score
            logger.debug(f"Document evaluation score: {score}")
            return score
        except Exception as e:
            logger.error(f"Error evaluating document: {str(e)}")
            # Default to mid-range score on error
            return 0.5
    
    def evaluate_documents(self, query: str, documents: List[str]) -> List[float]:
        """
        Evaluate multiple documents for relevance to a query.
        
        Args:
            query: The user's question
            documents: List of document contents
            
        Returns:
            List of relevance scores
        """
        return [self.evaluate_document(query, doc) for doc in documents]
    
    def refine_knowledge(self, document: str) -> List[str]:
        """
        Extract key information from a document.
        
        Args:
            document: The document content
            
        Returns:
            List of key points
        """
        prompt = PromptTemplate(
            input_variables=["document"],
            template=(
                "Extract the key information from the following document in bullet points:\n"
                "{document}\nKey points:"
            )
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
            result = chain.invoke({"document": document})
            key_points = [point.strip() for point in result.key_points.split('\n') if point.strip()]
            logger.debug(f"Extracted {len(key_points)} key points from document")
            return key_points
        except Exception as e:
            logger.error(f"Error refining knowledge: {str(e)}")
            return [document]  # Return original document on error
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query to make it more suitable for web search.
        
        Args:
            query: The original query
            
        Returns:
            Rewritten query
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Rewrite the following query to make it more suitable for a web search:\n"
                "{query}\nRewritten query:"
            )
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
            result = chain.invoke({"query": query})
            rewritten = result.query.strip()
            logger.info(f"Rewritten query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query  # Return original query on error
    
    @staticmethod
    def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
        """
        Parse web search results.
        
        Args:
            results_string: Raw search results as string
            
        Returns:
            List of (title, link) tuples
        """
        try:
            results = json.loads(results_string)
            return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
        except json.JSONDecodeError:
            logger.warning("Error parsing search results as JSON")
            return [("Web search results", "")]
    
    def perform_web_search(self, query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Perform a web search and refine the results.
        
        Args:
            query: The search query
            
        Returns:
            Tuple of (refined_knowledge, sources)
        """
        try:
            rewritten_query = self.rewrite_query(query)
            logger.info(f"Performing web search for: {rewritten_query}")
            
            web_results = self.search.run(rewritten_query)
            web_knowledge = self.refine_knowledge(web_results)
            sources = self.parse_search_results(web_results)
            
            logger.info(f"Web search returned {len(sources)} sources")
            return web_knowledge, sources
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return [f"Web search error: {str(e)}"], [("Web search", "")]
    
    def generate_answer(
        self, 
        query: str, 
        knowledge: str, 
        sources: List[Tuple[str, str]]
    ) -> str:
        """
        Generate an answer based on the query and knowledge.
        
        Args:
            query: The user's question
            knowledge: The compiled knowledge to answer from
            sources: List of (title, link) tuples
            
        Returns:
            Generated answer
        """
        # Format sources for the prompt
        sources_text = "\n".join([
            f"{title}: {link}" if link else title 
            for title, link in sources
        ])
        
        prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template=(
                "Based on the following knowledge, answer the query. "
                "Include the sources with their links (if available) at the end of your answer:\n\n"
                "Query: {query}\n\n"
                "Knowledge: {knowledge}\n\n"
                "Sources: {sources}\n\n"
                "Answer:"
            )
        )
        
        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "query": query,
                "knowledge": knowledge,
                "sources": sources_text
            })
            answer = result.content
            logger.info("Successfully generated answer")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def process_query(
        self, 
        query: str, 
        retrieved_docs: List[str],
        use_web_search: bool = False
    ) -> Tuple[str, List[str], str]:
        """
        Process a query using the CRAG approach.
        
        Args:
            query: The user's question
            retrieved_docs: Documents retrieved from vectorstore
            use_web_search: Whether to force web search
            
        Returns:
            Tuple of (answer, sources, action_taken)
            action_taken: 'correct', 'incorrect', 'ambiguous', or 'forced_web_search'
        """
        logger.info(f"Processing query with {len(retrieved_docs)} retrieved documents")
        
        # If web search is explicitly requested, skip evaluation
        if use_web_search:
            logger.info("Web search explicitly requested")
            web_knowledge, web_sources = self.perform_web_search(query)
            final_knowledge = "\n".join(web_knowledge)
            sources = [f"{title} ({link})" if link else title for title, link in web_sources]
            answer = self.generate_answer(query, final_knowledge, web_sources)
            return answer, sources, "forced_web_search"
        
        # Evaluate retrieved documents
        eval_scores = self.evaluate_documents(query, retrieved_docs)
        max_score = max(eval_scores) if eval_scores else 0
        
        logger.info(f"Evaluation scores: {eval_scores}, max: {max_score}")
        
        sources = []
        action_taken = ""
        
        # Determine action based on CRAG thresholds
        if max_score > self.upper_threshold:
            # Correct: Use retrieved document
            logger.info("Action: CORRECT - Using retrieved document")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            sources = ["Retrieved document from uploaded files"]
            action_taken = "correct"
            
        elif max_score < self.lower_threshold:
            # Incorrect: Perform web search
            logger.info("Action: INCORRECT - Performing web search")
            web_knowledge, web_sources = self.perform_web_search(query)
            final_knowledge = "\n".join(web_knowledge)
            sources = [f"{title} ({link})" if link else title for title, link in web_sources]
            action_taken = "incorrect"
            
        else:
            # Ambiguous: Combine retrieved document and web search
            logger.info("Action: AMBIGUOUS - Combining retrieved document and web search")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self.refine_knowledge(best_doc)
            web_knowledge, web_sources = self.perform_web_search(query)
            
            final_knowledge = "\n".join(["Retrieved information:"] + retrieved_knowledge + 
                                       ["\nWeb search information:"] + web_knowledge)
            
            sources = ["Retrieved document"] + [
                f"{title} ({link})" if link else title 
                for title, link in web_sources
            ]
            action_taken = "ambiguous"
        
        # Generate final answer
        answer = self.generate_answer(query, final_knowledge, [(s, "") for s in sources])
        
        return answer, sources, action_taken


# TODO: Integrate MCP (Model Context Protocol) server tools here as additional context providers.
# This could allow for additional data sources or specialized tools to be integrated
# into the retrieval and generation pipeline.

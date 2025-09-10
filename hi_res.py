import logging
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
import tempfile
import os
from langchain.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever
from db_utils import log_interaction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    
    def __init__(self):
        self.model = ChatOllama(model="llama3.1:8b")
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks on legal documents. 
            Use the provided context to answer the question concisely and accurately.

            - Extract ALL specific facts, including numbers (e.g., monetary amounts like $5 million), dates/timelines (e.g., September 5, 2024), and lists (e.g., all misrepresentations, all causes of action).
            - Structure answers with bullet points or sections for clarity: e.g., separate timelines, figures, and legal items.
            - If a fact spans multiple chunks, synthesize it completely without omission.
            - For questions about names or lists, extract ALL relevant information (e.g., proper nouns, bullet points) in a structured format.  
            - If no relevant information is found, state: "No relevant information found in the context."  
            - If the context is insufficient or incomplete, state: "Insufficient context to provide a complete answer."  
            - Never hallucinate or invent legal facts beyond the provided context. 
            - Always give well-formatted answers in bullet points that are precise, clear, and expanded where needed. 

            Question: {question}  

            Context: {context}  

            Answer:
            """
        )
        logger.info("ChatPDF initialized")

    def ingest(self, pdf_file_path: str):
        try:
            logger.info(f"Loading and partitioning PDF: {pdf_file_path}")
            elements = partition_pdf(
                filename=pdf_file_path,
                strategy="hi_res",
                model_name="yolox"
                # infer_table_structure=True,
                # extract_images_in_pdf=True
            )
            logger.info(f"Partitioned into {len(elements)} elements")

            # Perform smart chunking
            chunks = chunk_by_title(
                elements=elements,
                max_characters=1500,
                overlap = 400,
                combine_text_under_n_chars=75,
                new_after_n_chars=1200
            )
            logger.info(f"Chunked into {len(chunks)} semantic chunks")

            # Convert ElementMetadata to dictionary for compatibility
            processed_chunks = []
            for chunk in chunks:
                metadata = {}
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    # Convert ElementMetadata to dict
                    metadata_dict = chunk.metadata.to_dict() if hasattr(chunk.metadata, 'to_dict') else {}
                    # Filter to simple types (strings, numbers, etc.)
                    for key, value in metadata_dict.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)  # Convert complex types to string
                processed_chunks.append(Document(page_content=chunk.text, metadata=metadata))
            
            # Apply filter_complex_metadata
            processed_chunks = filter_complex_metadata(processed_chunks)
            
            # Create vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            self.vector_store = Chroma.from_documents(documents=processed_chunks, embedding=embeddings)
            logger.info("Vector store created")

            # Base retriever with higher k
            # base_retriever = self.vector_store.as_retriever(
            #     search_type="similarity_score_threshold",
            #     search_kwargs={
            #         "k": 6,
            #         "score_threshold": 0.35
            #     },
            # )
            # # keyword_retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
            # dense_retriever = self.vector_store.as_retriever(
            #     search_type="similarity",
            #     search_kwargs={"k": 12}
            # )
            
            # keyword_retriever = BM25Retriever.from_documents(processed_chunks)
            # keyword_retriever.k = 12


            # # Add re-ranker
            # compressor = LLMChainExtractor.from_llm(self.model)

            # compressed_dense = ContextualCompressionRetriever(
            #     base_compressor=compressor,
            #     base_retriever=dense_retriever
            # )
            # compressed_keyword = ContextualCompressionRetriever(
            #     base_compressor=compressor,
            #     base_retriever=keyword_retriever
            # )
            # # self.retriever = EnsembleRetriever(
            # #     retrievers=[compressed_dense, compressed_keyword],
            # #     weights=[0.5, 0.5]  # balance between embeddings & keyword search
            # # )
            # # Ensemble retriever
            # self.retriever = EnsembleRetriever(
            #     retrievers=[compressed_dense, compressed_keyword],  # Explicit list of retrievers
            #     weights=[0.6, 0.4]  # Matching number of weights to retrievers
            # )

            # Enhanced temporal-aware retriever setup
            similarity_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 25}  # Increased for better timeline coverage
            )

            # MMR for diversity (catches different types of content)  
            mmr_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 20,
                    "fetch_k": 60,  # Fetch many candidates
                    "lambda_mult": 0.5  # More diversity to get timeline + legal analysis
                }
            )

            # Keyword search (great for dates and names)
            keyword_retriever = BM25Retriever.from_documents(processed_chunks)
            keyword_retriever.k = 20

            # REMOVE COMPRESSION for better detail retention
            # Ensemble with rebalanced weights favoring semantic similarity
            self.retriever = EnsembleRetriever(
                retrievers=[similarity_retriever, mmr_retriever, keyword_retriever],
                weights=[0.5, 0.3, 0.2]  # Rebalanced for better fact coverage
            )


            logger.info("Hybrid ensemble retriever with compression initialized")
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            self.chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )
            logger.info("Chain created successfully")
        except Exception as e:
            logger.error(f"Error in ingest: {str(e)}")
            raise

    def ask(self, query: str):
        if not self.chain:
            logger.warning("No chain found, PDF not ingested")
            return "Please, add a PDF document first."

        logger.info(f"Processing query: {query}")
        retrieved_docs = self.retriever.get_relevant_documents(query)
        for i, doc in enumerate(retrieved_docs):
            logger.info(
                f"Retrieved chunk {i} (score: {doc.metadata.get('score', 'N/A')}): {doc.page_content[:1500]}..."
            )

        # get model answer
        answer = self.chain.invoke(query)

        # try to capture doc_name (from first retrieved doc's metadata)
        doc_name = None
        if retrieved_docs and "source" in retrieved_docs[0].metadata:
            doc_name = os.path.basename(retrieved_docs[0].metadata["source"])

        # log question + answer + doc_name into SQLite
        log_interaction(query, answer, doc_name)

        return answer


    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        logger.info("Session cleared")
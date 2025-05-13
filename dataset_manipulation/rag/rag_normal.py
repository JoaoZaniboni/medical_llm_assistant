### Loading the embedder
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import ServiceContext
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import qdrant_client
import yaml
import os
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from rag.pydantic_classes import Response, ResponseText
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank

from dotenv import load_dotenv
load_dotenv()

new_summary_tmpl_str = (
"Informações de contexto estão abaixo.\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"Dadas as informações de contexto e sem utilizar conhecimento prévio, "
"gere uma resposta para a query, em português do Brasil, e com a ortografia correta."
"Consulta: {query_str}\n"
"Resposta: "
)

class RAG:
    def __init__(self, llm, embedding_model, collection_name, qdrant_url):
        self.llm = llm  # ollama llm
        self.embed_model = embedding_model
        self.define_settings()
        self.aclient=qdrant_client.AsyncQdrantClient(url=qdrant_url)
        self.client = qdrant_client.QdrantClient(url=qdrant_url)
        self.qdrant_vector_store = QdrantVectorStore(
            client=self.client, aclient=self.aclient, collection_name= collection_name,
        )
        self.index = None
        if self.client.collection_exists(collection_name):
            self.index = VectorStoreIndex.from_vector_store(vector_store=self.qdrant_vector_store )
        self.storage_context = StorageContext.from_defaults(vector_store=self.qdrant_vector_store)
        self.new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    
    def define_settings(self):
         Settings.llm = self.llm
         Settings.embed_model = self.embed_model

    
    def create_engine(self, index, similarity_top_k, change_prompt = True):
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, 
                                            output=Response, response_mode="tree_summarize",
                                        verbose=True)
        if change_prompt:
            query_engine.update_prompts(
                {"response_synthesizer:summary_template": self.new_summary_tmpl}
            )
        return query_engine

    def query(self, query):
        if not self.index:
            return Response(
            search_result="Necessário adicioanar dados no DB antes de realizar query", source=[], chunks=[]
            )
        query_engine = self.create_engine(self.index, query.similarity_top_k)

        response = query_engine.query(query.query)
        response_object = Response(
            search_result=str(response).strip(), source=[response.metadata[k]["file_path"] for k in response.metadata.keys()],
                                                            chunks=[x.text for x in response.source_nodes])
    
        return response_object
    


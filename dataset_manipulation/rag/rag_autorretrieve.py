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
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from dotenv import load_dotenv
load_dotenv()

new_summary_tmpl_str = (
"Informações de contexto estão abaixo.\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"Imagignado ser um assistente de QA, dadas as informações de contexto e sem conhecimento prévio, "
"gere uma resposta para a consulta, sempre em português do Brasil, e com a ortografia correta. Se a resposta não puder ser formada \
estritamente usando o contexto, diga educadamente apenas que você \
não consegue responder a pergunta. \n"
"Consulta: {query_str}\n"
"Resposta: "
)

class RAG:
    def __init__(self, llm, embedding_model, chunk_size, similarity_top_k):
        self.llm = llm  # ollama llm
        self.aclient=qdrant_client.AsyncQdrantClient(url=os.getenv("qdrant_url"))
        self.client = qdrant_client.QdrantClient(url=os.getenv("qdrant_url"))
        self.qdrant_vector_store = QdrantVectorStore(
            client=self.client, aclient=self.aclient, collection_name=os.getenv('collection_name'),
        )


        self.embed_model = HuggingFaceEmbeddings(model_name=os.getenv('embedding_model'))
        self.node_parser = HierarchicalNodeParser.from_defaults()
        self.docstore = SimpleDocumentStore()
        self.storage_context = StorageContext.from_defaults(docstore=self.docstore, vector_store=self.qdrant_vector_store)
        # self.storage_context = StorageContext.from_defaults(vector_store=self.qdrant_vector_store)
        self.service_context = ServiceContext.from_defaults(
            llm= self.llm, embed_model=self.embed_model, chunk_size=int(os.getenv("chunk_size"))
        )
        self.index = None
        if self.client.collection_exists('digitro'):
            self.index = VectorStoreIndex.from_vector_store(vector_store=self.qdrant_vector_store,
                                                service_context=self.service_context,
                                                storage_context=self.storage_context, )
        
        
        # if self.index:
        #     self.index = None
        self.define_settings()


        self.new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    
    def define_settings(self):
         Settings.llm = self.llm
         Settings.embed_model = self.embed_model


    def ingest(self):
        print("Indexing data...")
        reader = SimpleDirectoryReader(os.getenv("data_path"))
        documents = reader.load_data()
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.storage_context.docstore.add_documents(nodes)
        leaf_nodes = get_leaf_nodes(nodes)
        self.index = VectorStoreIndex(
            leaf_nodes, storage_context=self.storage_context, service_context=self.service_context, show_progress=True
        )
        self.index.set_index_id("principe_index")
        response = ResponseText(response="Data inserido com sucesso"
        )
        return response
    
    def create_engine(self, similarity_top_k, change_prompt = True):
        automerging_retriever = self.index.as_retriever(
        similarity_top_k=similarity_top_k
        )

        retriever = AutoMergingRetriever(
            automerging_retriever, 
            self.index.storage_context, 
            verbose=True
        )

        self.rerank = SentenceTransformerRerank(top_n=similarity_top_k//2, model="BAAI/bge-reranker-base")

        query_engine = RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=[self.rerank]
        )
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
        
        query_engine = self.create_engine(query.similarity_top_k)

        response = query_engine.query(query.query)
        response_object = Response(
            search_result=str(response).strip(), source=[response.metadata[k]["file_path"] for k in response.metadata.keys()],
                                                            chunks=[x.text for x in response.source_nodes])
    
        return response_object
    


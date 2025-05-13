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

new_summary_tmpl_str = (
"Informações de contexto estão abaixo.\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"Imagignado ser um assistente sobre os produtos da empresa Dígitro, dadas as informações de contexto e sem conhecimento prévio, "
"gere uma resposta para a consulta. Se a resposta não puder ser formada \
estritamente usando o contexto, diga educadamente apenas que você \
não tem conhecimento sobre esse tópico. Responda em português do Brasil, e com a ortografia correta. \n"
"Consulta: {query_str}\n"
"Resposta: "
)

class RAG:
    def __init__(self, config_file, llm):
        self.config = config_file
        self.qdrant_client = qdrant_client.QdrantClient(
            url=self.config['qdrant_url']
        )
        self.llm = llm  # ollama llm
        
        self.client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        self.qdrant_vector_store = QdrantVectorStore(
            client=self.client, collection_name=self.config['collection_name']
        )
        self.embed_model = HuggingFaceEmbeddings(model_name=self.config['embedding_model'])
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        self.postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

        self.storage_context = StorageContext.from_defaults(vector_store=self.qdrant_vector_store)
        self.service_context = ServiceContext.from_defaults(
            llm= self.llm, embed_model=self.embed_model, chunk_size=self.config["chunk_size"]
        )
        self.index = None
        if self.client.collection_exists('digitro'):
            self.index = VectorStoreIndex.from_vector_store(vector_store=self.qdrant_vector_store,
                                                service_context=self.service_context,
                                                storage_context=self.storage_context)
        
        
        # if self.index:
        #     self.index = None
        self.define_settings()


        self.new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    
    def define_settings(self):
         Settings.llm = self.llm
         Settings.embed_model = self.embed_model


    def ingest(self):
        print("Indexing data...")
        reader = SimpleDirectoryReader(self.config["data_path"])
        documents = reader.load_data()
        nodes = self.node_parser.get_nodes_from_documents(documents)

        self.index = VectorStoreIndex(
            nodes, storage_context=self.storage_context, service_context=self.service_context
        )
        self.index.set_index_id("una_index")
        response = ResponseText(response="Data inserido com sucesso"
)
        return response
    
    def query(self, query):
        if not self.index:
            return Response(
            search_result="Necessário adicioanar dados no DB antes de realizar query", source=[], chunks=[]
            )
        query_engine = self.index.as_query_engine(similarity_top_k=query.similarity_top_k, 
                                                  output=Response, response_mode="tree_summarize",  node_postprocessors = [self.postproc],
                                                verbose=True, vector_store_query_mode="hybrid", alpha=0.5,)
        

        query_engine.update_prompts(
            {"response_synthesizer:summary_template": self.new_summary_tmpl}
        )

        response = query_engine.query(query.query)
        response_object = Response(
            search_result=str(response).strip(), source=[response.metadata[k]["file_path"] for k in response.metadata.keys()],
                                                            chunks=[x.text for x in response.source_nodes])
    
        return response_object
    

    # def _load_config(self, config_file):
    #     with open(config_file, "r") as stream:
    #         try:
    #             self.config = yaml.safe_load(stream)
    #         except yaml.YAMLError as e:
    #             print(e)


import os
from llama_index.core.llama_dataset import LabelledRagDataset
from evaluator_class import RagEvaluatorPack
import json
import pandas as pd
import os
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core import Document
from rag.rag_normal import RAG
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


llm_judge = Ollama(model="llama3.1", request_timeout=120.0)
llm = Ollama(model="llama3.1", request_timeout=120.0)
embedding_model = "sentence-transformers/all-mpnet-base-v2"
collection_name = "normal_rag_test"
qdrant_url = "http://localhost:6333"


embedding = HuggingFaceEmbeddings(model_name=embedding_model)
rag = RAG(llm, embedding, collection_name, qdrant_url)

similarity_top_k = 5
# Caminho para o arquivo JSON
caminho_arquivo = 'dataset_manipulation/dataset_test.json'

# Lê o arquivo JSON
with open(caminho_arquivo, 'r', encoding='utf-8') as f:
    dados = json.load(f)

# Gera uma lista com as conversas concatenadas
conversas = [TextNode(text='\n'.join(item['utterances'])) for item in dados]


if not rag.client.collection_exists(collection_name):
    rag.index = VectorStoreIndex(conversas, embed_model=embedding, storage_context=rag.storage_context, show_progress=True)
    rag.index.set_index_id("test_index")

query_engine = rag.create_engine(rag.index, similarity_top_k)
# if os.path.exists('local-rag-llamaindex/pack/llama_index/packs/rag_evaluator/base.py'):
    
rag_dataset = LabelledRagDataset.from_json("dataset_manipulation/eval_ds.json")

rag_evaluator = RagEvaluatorPack(
    query_engine= query_engine,  # built with the same source Documents as the rag_dataset
    rag_dataset=rag_dataset,
    judge_llm= llm_judge
)

df_result = rag_evaluator.run()
print(df_result)
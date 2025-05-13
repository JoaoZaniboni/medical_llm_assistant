import json
import pandas as pd
import os

from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core import Document


llm = Ollama(model="llama3.1", request_timeout=120.0)


text_question_template= PromptTemplate("""
Conversas entre pacientes e médicos de exemplo estão abaixo.
---------------------
{context_str}
---------------------
Dadas as informações de contexto,
gere exatamente uma pergunta.
A pergunta deve ser relacionada ao que o paciente do contexto perguntou, e ao que o médico respondeu. 
Responda diretamente com a pergunta, 
sem qualquer texto adicional. Utilize apenas Português do Brasil.
{query_str}""")

text_qa_template = PromptTemplate(
"Conversas entre pacientes e médicos de exemplo estão abaixo."
"---------------------"
"{context_str}"
"---------------------"
"Utilizando Português do Brasil sempre e ortografia correta,"
"dadas as informações, responda à consulta."
"Consulta: {query_str}"
"Resposta:"
)

question_gen_query = f""" Você é um Paciente. Sua tarefa é elaborar \
1 pergunta para uma doutor. \
A pergunta deve ser relacionada ao que o paciente do contexto perguntou, e ao que o médico respondeu.
Faça a pergunta sem qualquer texto adicional.
Não inclua introduções, numerações ou qualquer outro texto. \
Use ponto de interrogação no final de cada pergunta. Responda exatamente neste formato:
\n
Pergunta
\n
Quero que utilize Português do Brasil sempre. Lembre-se que deve apenas gerar 1 pergunta diretamente, usando português."""


# Caminho para o arquivo JSON
caminho_arquivo = 'dataset_manipulation/dataset_test.json'

# Lê o arquivo JSON
with open(caminho_arquivo, 'r', encoding='utf-8') as f:
    dados = json.load(f)

# Gera uma lista com as conversas concatenadas
conversas = [Document(text='\n'.join(item['utterances'])) for item in dados]

num_questions_per_chunk = 1

dataset_generator = RagDatasetGenerator.from_documents(
documents=conversas,
llm=llm,
num_questions_per_chunk=num_questions_per_chunk,  # set the number of questions per nodes
text_qa_template = text_qa_template,
question_gen_query = question_gen_query,
text_question_template = text_question_template,
show_progress = True
)
# RagDatasetGenerator.update_prompts()

qa_dataset = dataset_generator.generate_dataset_from_nodes()
qa_dataset.save_json("dataset_manipulation/eval_ds.json")
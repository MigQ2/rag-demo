# Databricks notebook source
# MAGIC %md ## Install required packages

# COMMAND ----------

# %pip install databricks-sdk --upgrade

# # %pip install --force-reinstall databricks-feature-engineering
# %pip install --force-reinstall databricks-vectorsearch 
# %pip install --force-reinstall -v langchain openai


# COMMAND ----------

# %restart_python

# COMMAND ----------

catalog_name = "rag_poc"
schema_name = "contracts_rag"

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import mlflow 
from databricks import feature_engineering
import json
import requests
import time


# COMMAND ----------


fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md ### Create a vector index for unstructured data searches
# MAGIC
# MAGIC Databricks vector search allows you to ingest and query unstructured data. 

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vector_search_endpoint_name = "rag_endpoint_v2"

try:
    vsc.create_endpoint(vector_search_endpoint_name)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

vsc.list_endpoints()
vsc.list_indexes(vector_search_endpoint_name)

# COMMAND ----------

def vector_index_ready(vsc, endpoint_name, index_name, desired_state_prefix="ONLINE"):
    index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name).describe()
    status = index["status"]["detailed_state"]
    return desired_state_prefix in status


def vector_index_exists(vsc, endpoint_name, index_name):
    try:
        vsc.get_index(endpoint_name=endpoint_name, index_name=index_name).describe()
        return True
    except Exception as e:
        if "DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            return False
        else:
            raise e

def wait_for_vector_index_ready(vsc, endpoint_name, index_name, max_wait_time=2400, desired_state_prefix="ONLINE"):
    wait_interval = 60
    max_wait_intervals = int(max_wait_time / wait_interval)
    for i in range(0, max_wait_intervals):
        time.sleep(wait_interval)
        if vector_index_ready(vsc, endpoint_name, index_name, desired_state_prefix):
            print(f"Vector search index '{index_name}' is ready.")
            return
        else:
            print(
                f"Waiting for vector search index '{index_name}' to be in ready state."
            )
    raise Exception(f"Vector index '{index_name}' is not ready after {max_wait_time} seconds.")

# COMMAND ----------

# MAGIC %md ### Calculate embedding using Databricks foundational model 

# COMMAND ----------

def calculate_embedding(text):
    embedding_endpoint_name = "databricks-bge-large-en"
    url = f"https://{mlflow.utils.databricks_utils.get_browser_hostname()}/serving-endpoints/{embedding_endpoint_name}/invocations"
    databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token

    headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
        
    data = {
        "input": text
    }
    data_json = json.dumps(data, allow_nan=True)
    
    print(f"\nCalling Embedding Endpoint: {embedding_endpoint_name}\n")
    
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()['data'][0]['embedding']

# COMMAND ----------

# MAGIC %md ### Create feature table for contract clauses

# COMMAND ----------

contracts_table = f"{catalog_name}.{schema_name}.contract_clauses"

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {contracts_table} (
    clause_id STRING,
    clause_text STRING,
    embedding ARRAY<DOUBLE>,
    CONSTRAINT clause_id PRIMARY KEY(clause_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true);
""")

# COMMAND ----------

schema = StructType([
    # StructField("clause_id", StringType(), False),
    StructField("clause_text", StringType(), False),
    StructField("embedding", ArrayType(DoubleType()), False)
])

# Load the JSON file
with open('../data/contract_sections.json', 'r') as f:
    loaded_sections = json.load(f)

loaded_sections


# COMMAND ----------

import pyspark.sql.functions as F

# Create some dummy embeddings
data = [(s.replace("\n", ""), calculate_embedding(s)) for s in loaded_sections]

# Create a DataFrame with the dummy data
df = spark.createDataFrame(data, schema=schema)

df = df.withColumn("clause_id", F.expr("AI_SUMMARIZE(clause_text)"))


# Create the feature table that holds the embeddings of the hotel characteristics
fe.write_table(
    name=contracts_table, df=df.select("clause_id", "clause_text", "embedding")
)

# COMMAND ----------

# MAGIC %md ### Setup Vector Search Index

# COMMAND ----------

# MAGIC %md #### Create a vector search index based on the embeddings feature table

# COMMAND ----------

contract_table_index = f"{catalog_name}.{schema_name}.contract_index"

try:
  vsc.create_delta_sync_index(
      endpoint_name=vector_search_endpoint_name,
      index_name=contract_table_index,
      source_table_name=contracts_table,
      pipeline_type="TRIGGERED",
      primary_key="clause_id",
      embedding_dimension=1024, # Match your model embedding size (bge)
      embedding_vector_column="embedding"
  )
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# MAGIC %md #### Wait for the vector search index to be ready

# COMMAND ----------

vector_index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=contract_table_index)
vector_index.wait_until_ready(verbose=True)

# COMMAND ----------

# MAGIC %md ## AI Bot powered by Databricks Feature Serving and Databricks online tables
# MAGIC 1. Automatically sync data from Delta table to online table.
# MAGIC 1. Lookup features in real-time with low latency.
# MAGIC 1. Provide context and augment chatbots with enterprise data as shown in this example.
# MAGIC 1. Implement best practices of data management in MLOps with LLMOps.

# COMMAND ----------

from langchain.tools import BaseTool
from typing import Union, List
from databricks.vector_search.client import VectorSearchClient

class ContractClauseRetrievalTool(BaseTool):
    name: str = "Clausulas del contrato relevantes para la pregunta"
    description: str = "Utiliza esta tool cuando necesites encontrar las cláusulas del contrato que responden una pregunta acerca de las condiciones de la tarjeta de crédito"

    def _run(self, user_question: str):
        def calculate_embedding(text):
            embedding_endpoint_name = "databricks-bge-large-en"
            url = f"https://{mlflow.utils.databricks_utils.get_browser_hostname()}/serving-endpoints/{embedding_endpoint_name}/invocations"
            databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token

            headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
                
            data = {
                "input": text
            }
            data_json = json.dumps(data, allow_nan=True)
            
            print(f"\nCalling Embedding Endpoint: {embedding_endpoint_name}\n")
            
            response = requests.request(method='POST', headers=headers, url=url, data=data_json)
            if response.status_code != 200:
                raise Exception(f'Request failed with status {response.status_code}, {response.text}')

            return response.json()['data'][0]['embedding']
            
        try:
            vsc = VectorSearchClient()
            index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=contract_table_index)
            print(index)
            resp = index.similarity_search(columns=["clause_id", "clause_text"], query_vector=calculate_embedding(user_question), num_results=5, filters={})
            print(resp)
            data_array = resp and resp.get('result', {}).get('data_array')
            print(data_array)
        except Exception as e:
            print(f"Exception while running test case {e}")
            return []

        result = [contract[1] for contract in data_array]
        print(result)
        return result
    
    def _arun(self, user_id: str):
        raise NotImplementedError("This tool does not support async")

# COMMAND ----------

user_question = """Como efectuo el pago de la tarjeta?"""

def calculate_embedding(text):
    embedding_endpoint_name = "databricks-bge-large-en"
    url = f"https://{mlflow.utils.databricks_utils.get_browser_hostname()}/serving-endpoints/{embedding_endpoint_name}/invocations"
    databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token

    headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
        
    data = {
        "input": text
    }
    data_json = json.dumps(data, allow_nan=True)
    
    print(f"\nCalling Embedding Endpoint: {embedding_endpoint_name}\n")
    
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()['data'][0]['embedding']
    

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=contract_table_index)
print(index)
resp = index.similarity_search(columns=["clause_id", "clause_text"], query_vector=calculate_embedding(user_question), num_results=5, filters={})
print(resp)
data_array = resp and resp.get('result', {}).get('data_array')
print(data_array)
result = [contract[1] for contract in data_array]
print(result)







# COMMAND ----------

import pandas as pd

# Assuming data_array is already defined
df = pd.DataFrame(data_array, columns=['clause_id', "clause_text", 'similarity'])
display(df)

# COMMAND ----------

from langchain.agents import initialize_agent
# Tool imports
from langchain.agents import Tool

tools = [
  ContractClauseRetrievalTool(),
]
from databricks_langchain import ChatDatabricks
from langchain.llms import Databricks
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Initialize LLM (using Databricks built-in model)
# llm = Databricks(
#     endpoint_name='databricks-meta-llama-3-1-8b-instruct',
#     temperature=0
# )

llm = ChatDatabricks(
    endpoint='databricks-llama-4-maverick',
    temperature=0
)

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

# Initialize agent with tools
aibot = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=5,
    early_stopping_method='force',
    memory=conversational_memory
)

# COMMAND ----------

sys_msg = """Eres un asistente virtual para un banco cuyo objetivo es ayudar a los clientes a resolver preguntas sobre su contrato de tarjeta de crédito.

Si el usuario pregunta sobre cualquier tema diferente o no relacionado a su contrato de tarjeta de crédito, indica que no puedes ayudar con eso ya que no es tu función y que se limite a preguntar acerca de su tarjeta de crédito".

Utiliza la tool de retrieval para encontrar las cláusulas del contrato que responden a la pregunta del usuario.

Es importante que al responder una pregunta, identifiques la cláusula que contiene la información que responde la pregunta del usuario. 

Como el lenguaje de las cláusulas del contrato es muy formal, debes pensar en dar una respuesta que sea entendible por una persona sin mucho conocimiento legal o financiero, reinterpretando el contenido de la cláusula, que será muy formal, para que el usuario pueda entender tu respuesta. Adicionalmente, al final de tu respuesta incluye el texto literal de la cláusula o cláusulas que responden la pregunta para que el usuario pueda leerlas literalmente si desea

"""

# COMMAND ----------

new_prompt = aibot.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

aibot.agent.llm_chain.prompt = new_prompt

# COMMAND ----------

# MAGIC %md
# MAGIC By incorporating context from the Databricks Lakehouse including online tables and a feature serving endpoint, an AI chatbot created with context retrieval tools performs much better than a generic chatbot. 

# COMMAND ----------

aibot_output = aibot('Como hago el pago de mi tarjeta?')

# COMMAND ----------

print(aibot_output['output'])

# COMMAND ----------

from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint


resources = [
  DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
  DatabricksFunction(function_name=tool.uc_function_name),
]


with mlflow.start_run():
  logged_agent_info = mlflow.pyfunc.log_model(
    artifact_path="agent",
    python_model="agent.py",
    pip_requirements=[
      "mlflow",
      "langchain",
      "langgraph",
      "databricks-langchain",
      "unitycatalog-langchain[databricks]",
      "pydantic",
    ],
    resources=resources,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleanup
# MAGIC Uncomment lines 2 - 5 in the following cell to clean up the endpoints created in this notebook.

# COMMAND ----------

# Delete endpoint
# status = fs.delete_feature_serving_endpoint(name=user_endpoint_name)
# print(status)
# status = fs.delete_feature_serving_endpoint(name=hotel_endpoint_name)
# print(status)


# Cleanup for online table from Unity Catalog
